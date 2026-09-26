#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.execute_primary_analysis import validate_plan_provenance  # noqa: E402
from trajuq.aggregation import (  # noqa: E402
    PredictionCell,
    hierarchical_macro_bootstrap,
    metric_value,
    oriented_improvement,
)
from trajuq.features import FROZEN_FEATURE_GROUPS  # noqa: E402
from trajuq.io import read_jsonl  # noqa: E402


METRICS = ("auroc", "auprc", "aurc")
BASELINE = "decision_only"


def load_cell(spec: dict) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    label = str(spec["cell"])
    pred_path = Path(str(spec["outputs"]["feature_ablation_predictions"]))
    target_path = Path(str(spec["inputs"]["test"]["path"]))

    rows = list(read_jsonl(pred_path))
    target_rows = list(read_jsonl(target_path))
    if not rows:
        raise SystemExit(f"{label}: empty ablation prediction artifact")
    if not target_rows:
        raise SystemExit(f"{label}: empty frozen test target")

    ids = [str(row.get("example_id", "")) for row in rows]
    target_ids = [str(row.get("example_id", "")) for row in target_rows]
    if (
        any(not item_id for item_id in ids)
        or any(not item_id for item_id in target_ids)
        or len(ids) != len(set(ids))
        or len(target_ids) != len(set(target_ids))
        or ids != target_ids
    ):
        raise SystemExit(
            f"{label}: ablation predictions do not exactly cover the frozen test target"
        )

    expected_groups = list(FROZEN_FEATURE_GROUPS)
    target_by_id = {str(row["example_id"]): row for row in target_rows}
    y_error: list[int] = []
    scores: dict[str, list[float]] = {name: [] for name in expected_groups}

    for row in rows:
        item_id = str(row["example_id"])
        target = target_by_id[item_id]
        expected_y = 0 if bool(target["correct"]) else 1
        if int(row.get("y_error", -1)) != expected_y:
            raise SystemExit(f"{label}: ablation label mismatch for {item_id}")
        if bool(row.get("correct")) != bool(target["correct"]):
            raise SystemExit(f"{label}: ablation correctness mismatch for {item_id}")

        group_scores = row.get("p_error_by_group", {})
        if set(group_scores) != set(expected_groups):
            raise SystemExit(
                f"{label}: ablation group set drifted: "
                f"{sorted(group_scores)} != {sorted(expected_groups)}"
            )
        y_error.append(expected_y)
        for name in expected_groups:
            value = float(group_scores[name])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SystemExit(
                    f"{label}: invalid calibrated probability {name}={value!r}"
                )
            scores[name].append(value)

    return (
        ids,
        np.asarray(y_error, dtype=int),
        {
            name: np.asarray(values, dtype=float)
            for name, values in scores.items()
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hierarchical equal-cell-weight aggregation for the frozen "
            "feature-family ablations."
        )
    )
    parser.add_argument(
        "--plan",
        default="results/primary_analysis_plan.json",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--output",
        default="results/primary_analysis/macro_feature_ablation.json",
    )
    args = parser.parse_args()

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("protocol") != "trajuq_primary_analysis_plan_v2":
        raise SystemExit("Primary analysis plan protocol is not v2.")
    if plan.get("status") != "PASS":
        raise SystemExit("Primary analysis plan is not PASS.")

    failures = validate_plan_provenance(plan)
    if failures:
        raise SystemExit(
            "Primary analysis provenance validation failed:\n- "
            + "\n- ".join(failures)
        )

    specs = list(plan.get("cell_analyses", []))
    expected = int(plan.get("cell_count", 0))
    if expected <= 0 or len(specs) != expected:
        raise SystemExit(
            f"Ablation plan cell count mismatch: {len(specs)} != {expected}"
        )

    pairs = [(str(spec["adapter"]), str(spec["dataset"])) for spec in specs]
    if len(pairs) != len(set(pairs)):
        raise SystemExit("Ablation plan contains duplicate model-dataset cells.")

    loaded: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for spec in specs:
        _, y_error, scores = load_cell(spec)
        loaded[str(spec["cell"])] = (y_error, scores)

    candidate_groups = [
        name for name in FROZEN_FEATURE_GROUPS
        if name != BASELINE
    ]
    group_results: dict[str, dict] = {}
    per_cell: dict[str, dict] = {}

    for spec in specs:
        label = str(spec["cell"])
        y_error, scores = loaded[label]
        baseline = scores[BASELINE]
        per_cell[label] = {
            "adapter": str(spec["adapter"]),
            "dataset": str(spec["dataset"]),
            "n": int(len(y_error)),
            "groups": {},
        }
        for group in candidate_groups:
            candidate = scores[group]
            per_cell[label]["groups"][group] = {
                metric: {
                    "candidate": metric_value(y_error, candidate, metric),
                    "baseline": metric_value(y_error, baseline, metric),
                    "improvement": oriented_improvement(
                        y_error,
                        candidate,
                        baseline,
                        metric,
                    ),
                }
                for metric in METRICS
            }

    for group in candidate_groups:
        cells: list[PredictionCell] = []
        for spec in specs:
            label = str(spec["cell"])
            y_error, scores = loaded[label]
            cells.append(
                PredictionCell(
                    label=label,
                    y_error=y_error,
                    candidate_score=scores[group],
                    baseline_score=scores[BASELINE],
                    model_family=str(spec["adapter"]),
                    dataset=str(spec["dataset"]),
                )
            )
        group_results[group] = {
            "features": list(FROZEN_FEATURE_GROUPS[group]),
            "metrics": {
                metric: hierarchical_macro_bootstrap(
                    cells,
                    metric=metric,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
                for metric in METRICS
            },
        }

    payload = {
        "protocol": "trajuq_feature_ablation_macro_v1",
        "baseline": BASELINE,
        "delta_direction": "positive_means_group_is_better_than_decision_only",
        "plan": str(plan_path),
        "expected_cells": expected,
        "observed_cells": len(specs),
        "cell_set_complete": len(specs) == expected,
        "group_order": list(FROZEN_FEATURE_GROUPS),
        "candidate_group_order": candidate_groups,
        "frozen_groups": {
            name: list(features)
            for name, features in FROZEN_FEATURE_GROUPS.items()
        },
        "per_cell": per_cell,
        "groups": group_results,
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
