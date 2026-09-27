#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.execute_primary_analysis import (  # noqa: E402
    semantic_output_valid,
    validate_plan_provenance,
)
from trajuq.provenance import analysis_provenance  # noqa: E402


METRICS = (
    "brier",
    "nll",
    "ece",
    "ece_10",
    "ece_20",
    "aurc",
    "risk_at_50_coverage",
    "risk_at_80_coverage",
)
VIEWS = ("decision_entropy", "trajectory_plus_decision")


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite sensitivity metric")
    return float(arr.mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the frozen isotonic calibration sensitivity across the "
            "complete primary cell set. Ineligible cells remain explicit skips."
        )
    )
    parser.add_argument(
        "--plan",
        default="results/primary_analysis_plan.json",
    )
    parser.add_argument(
        "--output",
        default="results/primary_analysis/macro_calibration_sensitivity.json",
    )
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.is_file():
        raise SystemExit(f"Missing primary analysis plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    failures = validate_plan_provenance(plan)
    if failures:
        raise SystemExit(
            "Primary analysis plan provenance failed:\n- "
            + "\n- ".join(failures)
        )

    cells = list(plan.get("cell_analyses", []))
    expected = int(plan.get("cell_count", 0))
    if expected <= 0 or len(cells) != expected:
        raise SystemExit(
            f"Analysis plan cell count mismatch: {len(cells)} != {expected}"
        )

    sources: dict[str, Path] = {"analysis_plan": plan_path}
    eligible_labels: list[str] = []
    skipped_labels: list[str] = []
    per_cell: dict[str, dict] = {}
    collected: dict[str, dict[str, dict[str, list[float]]]] = {
        view: {
            metric: {
                "primary_logistic": [],
                "isotonic_sensitivity": [],
                "isotonic_minus_logistic_improvement": [],
            }
            for metric in METRICS
        }
        for view in VIEWS
    }

    for cell in cells:
        label = str(cell["cell"])
        path = Path(str(cell["outputs"]["calibration_sensitivity"]))
        if not semantic_output_valid(cell, "calibration_sensitivity", path):
            raise SystemExit(
                f"{label}: calibration sensitivity artifact is missing, stale, "
                "or violates the frozen eligibility/method contract"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources[f"cell:{label}"] = path

        status = str(payload.get("status", ""))
        eligibility = payload.get("eligibility", {})
        row = {
            "status": status,
            "eligible": bool(eligibility.get("eligible")),
            "n_calibration": int(payload.get("n_calibration", 0)),
            "n_test": int(payload.get("n_test", 0)),
            "observed_class_counts": eligibility.get(
                "observed_class_counts",
                {},
            ),
        }
        if status == "PASS":
            eligible_labels.append(label)
            for view in VIEWS:
                node = payload[view]
                for metric in METRICS:
                    collected[view][metric]["primary_logistic"].append(
                        float(node["primary_logistic"][metric])
                    )
                    collected[view][metric]["isotonic_sensitivity"].append(
                        float(node["isotonic_sensitivity"][metric])
                    )
                    collected[view][metric][
                        "isotonic_minus_logistic_improvement"
                    ].append(
                        float(
                            node[
                                "isotonic_minus_logistic_improvement"
                            ][metric]
                        )
                    )
        elif status == "SKIP_INSUFFICIENT_CALIBRATION":
            skipped_labels.append(label)
        else:
            raise SystemExit(f"{label}: unexpected sensitivity status {status!r}")
        per_cell[label] = row

    macro: dict[str, dict] = {}
    for view in VIEWS:
        macro[view] = {}
        for metric in METRICS:
            node = collected[view][metric]
            macro[view][metric] = {
                key: mean_or_none(values)
                for key, values in node.items()
            }

    payload = {
        "protocol": "trajuq_isotonic_calibration_sensitivity_macro_v1",
        "role": "supplementary_calibration_sensitivity",
        "provenance": analysis_provenance(sources),
        "expected_cells": expected,
        "observed_cells": len(cells),
        "cell_set_complete": len(cells) == expected,
        "eligibility_policy": (
            cells[0].get("calibration_sensitivity_contract", {})
            if cells
            else {}
        ),
        "eligible_cell_count": len(eligible_labels),
        "skipped_cell_count": len(skipped_labels),
        "eligible_cells": eligible_labels,
        "skipped_cells": skipped_labels,
        "aggregation": (
            "equal_weight_over_pre_registered_eligible_cells_only; all frozen "
            "ineligible cells remain explicit skips"
        ),
        "delta_direction": (
            "positive means isotonic improves a lower-is-better metric "
            "relative to the primary logistic calibrator"
        ),
        "metrics": list(METRICS),
        "per_cell": per_cell,
        "macro_over_eligible_cells": macro,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        existing = out.read_text(encoding="utf-8")
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if existing != rendered:
            raise SystemExit(
                f"Refusing to overwrite differing frozen sensitivity macro: {out}"
            )
        print(f"[TrajUQ] existing sensitivity macro is byte-identical: {out}")
        return
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
