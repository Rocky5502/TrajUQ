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
from trajuq.aggregation import (  # noqa: E402
    PredictionCell,
    hierarchical_macro_bootstrap,
    metric_value,
    oriented_improvement,
)
from trajuq.io import read_jsonl  # noqa: E402
from trajuq.provenance import analysis_provenance  # noqa: E402


SIGNALS = {
    "trajectory_entropy": "p_error_trajectory_entropy",
    "top1_revision_rate": "p_error_top1_revision_rate",
}
METRICS = ("auroc", "auprc", "aurc")


def exact_ids(path: Path) -> list[str]:
    rows = list(read_jsonl(path))
    ids = [str(row.get("example_id", "")) for row in rows]
    if (
        not rows
        or any(not item_id for item_id in ids)
        or len(ids) != len(set(ids))
    ):
        raise SystemExit(f"Invalid example IDs in {path}")
    return ids


def load_cell(
    spec: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    label = str(spec["cell"])
    trajuq_path = Path(str(spec["outputs"]["predictions"]))
    prior_path = Path(
        str(spec["outputs"]["prior_signal_predictions"])
    )

    if not semantic_output_valid(spec, "predictions", trajuq_path):
        raise SystemExit(
            f"{label}: TrajUQ prediction artifact is stale or not "
            "source-bound to the frozen analysis inputs"
        )
    if not semantic_output_valid(
        spec,
        "prior_signal_predictions",
        prior_path,
    ):
        raise SystemExit(
            f"{label}: prior-signal prediction artifact is stale or not "
            "source-bound to the frozen analysis inputs"
        )

    trajuq_rows = list(read_jsonl(trajuq_path))
    prior_rows = list(read_jsonl(prior_path))
    trajuq_ids = exact_ids(trajuq_path)
    prior_ids = exact_ids(prior_path)
    if trajuq_ids != prior_ids:
        raise SystemExit(
            f"{label}: prior-signal predictions do not exactly match "
            "TrajUQ prediction IDs"
        )

    y_error = np.asarray(
        [int(row["y_error"]) for row in trajuq_rows],
        dtype=int,
    )
    prior_y = np.asarray(
        [int(row["y_error"]) for row in prior_rows],
        dtype=int,
    )
    if not np.array_equal(y_error, prior_y):
        raise SystemExit(
            f"{label}: prior-signal labels differ from TrajUQ labels"
        )

    trajuq_score = np.asarray(
        [float(row["p_error_trajuq"]) for row in trajuq_rows],
        dtype=float,
    )
    if not np.all(np.isfinite(trajuq_score)):
        raise SystemExit(f"{label}: non-finite TrajUQ probabilities")

    prior_scores: dict[str, np.ndarray] = {}
    for signal, field in SIGNALS.items():
        values = np.asarray(
            [float(row[field]) for row in prior_rows],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise SystemExit(
                f"{label}: non-finite prior-signal probabilities "
                f"for {signal}"
            )
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise SystemExit(
                f"{label}: prior-signal probabilities outside [0,1] "
                f"for {signal}"
            )
        prior_scores[signal] = values

    return y_error, trajuq_score, prior_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Macro-aggregate prior-aligned single-trajectory signal "
            "diagnostics against TrajUQ on the complete frozen 18-cell set."
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
        default=(
            "results/primary_analysis/"
            "macro_prior_signal_diagnostics.json"
        ),
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
    if expected != 18 or len(specs) != expected:
        raise SystemExit(
            f"Prior-signal macro requires 18 cells; got {len(specs)}."
        )

    loaded = {
        str(spec["cell"]): load_cell(spec)
        for spec in specs
    }

    per_cell: dict[str, dict] = {}
    for spec in specs:
        label = str(spec["cell"])
        y_error, trajuq_score, prior_scores = loaded[label]
        per_cell[label] = {
            "adapter": str(spec["adapter"]),
            "dataset": str(spec["dataset"]),
            "n": int(len(y_error)),
            "signals": {},
        }
        for signal, score in prior_scores.items():
            per_cell[label]["signals"][signal] = {
                metric: {
                    "prior_signal": metric_value(
                        y_error,
                        score,
                        metric,
                    ),
                    "trajuq": metric_value(
                        y_error,
                        trajuq_score,
                        metric,
                    ),
                    "trajuq_improvement": oriented_improvement(
                        y_error,
                        trajuq_score,
                        score,
                        metric,
                    ),
                }
                for metric in METRICS
            }

    macro: dict[str, dict] = {}
    for signal in SIGNALS:
        cells: list[PredictionCell] = []
        for spec in specs:
            label = str(spec["cell"])
            y_error, trajuq_score, prior_scores = loaded[label]
            cells.append(
                PredictionCell(
                    label=label,
                    y_error=y_error,
                    candidate_score=trajuq_score,
                    baseline_score=prior_scores[signal],
                    model_family=str(spec["adapter"]),
                    dataset=str(spec["dataset"]),
                )
            )
        macro[signal] = {
            metric: hierarchical_macro_bootstrap(
                cells,
                metric=metric,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            for metric in METRICS
        }

    provenance_sources: dict[str, Path] = {}
    for spec in specs:
        label = str(spec["cell"])
        provenance_sources[f"trajuq_prediction:{label}"] = Path(
            str(spec["outputs"]["predictions"])
        )
        provenance_sources[f"prior_signal_prediction:{label}"] = Path(
            str(spec["outputs"]["prior_signal_predictions"])
        )

    payload = {
        "protocol": "trajuq_prior_aligned_signal_macro_v1",
        "provenance": analysis_provenance(provenance_sources),
        "role": "secondary_prior_aligned_signal_class_diagnostic",
        "comparison": "trajuq_vs_prior_aligned_single_trajectory_signals",
        "delta_direction": "positive_means_trajuq_is_better",
        "not_a_reproduction_of": [
            "D-CoCoA-L",
            "D-CoCoA-G",
            "DiSE",
        ],
        "plan": str(plan_path),
        "expected_cells": expected,
        "observed_cells": len(specs),
        "cell_set_complete": len(specs) == expected,
        "signal_order": list(SIGNALS),
        "per_cell": per_cell,
        "macro": macro,
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
