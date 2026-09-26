#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajuq.aggregation import (
    PredictionCell,
    hierarchical_macro_bootstrap,
    metric_value,
    oriented_improvement,
)
from trajuq.io import read_jsonl


AGGREGATE_METRICS = (
    "auroc",
    "auprc",
    "brier",
    "nll",
    "ece15",
    "aurc",
    "risk50",
    "risk80",
)


def index(path: str) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    out = {str(row["example_id"]): row for row in rows}
    if not rows:
        raise SystemExit(f"Empty prediction file: {path}")
    if len(out) != len(rows):
        raise SystemExit(f"Duplicate prediction IDs: {path}")
    return out


def load_cell(spec: dict) -> PredictionCell:
    label = str(spec["cell"])
    traj = index(str(spec["trajuq_predictions"]))
    repeated = index(str(spec["repeated_predictions"]))
    if set(traj) != set(repeated):
        raise SystemExit(f"{label}: prediction ID sets differ")

    ids = sorted(traj)
    y = np.asarray([int(traj[item]["y_error"]) for item in ids], dtype=int)
    y_repeat = np.asarray(
        [int(repeated[item]["y_error"]) for item in ids],
        dtype=int,
    )
    if not np.array_equal(y, y_repeat):
        raise SystemExit(f"{label}: correctness labels differ")

    candidate = np.asarray(
        [float(traj[item]["p_error_trajuq"]) for item in ids],
        dtype=float,
    )
    baseline = np.asarray(
        [float(repeated[item]["p_error_repeated_sampling"]) for item in ids],
        dtype=float,
    )
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(baseline)):
        raise SystemExit(f"{label}: non-finite probabilities")

    return PredictionCell(
        label=label,
        y_error=y,
        candidate_score=candidate,
        baseline_score=baseline,
        model_family=str(spec["adapter"]),
        dataset=str(spec["dataset"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Macro-aggregate the complete matched TrajUQ versus five-sample "
            "baseline subset defined by a frozen repeated-sampling plan."
        )
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if plan.get("status") != "PASS":
        raise SystemExit("Repeated-sampling plan is not PASS.")
    specs = plan.get("cells", [])
    expected = int(plan.get("expected_cells", 0))
    if expected <= 0 or len(specs) != expected:
        raise SystemExit(
            f"Repeated-sampling plan cell count mismatch: {len(specs)} != {expected}"
        )

    pairs = [(str(item["adapter"]), str(item["dataset"])) for item in specs]
    if len(pairs) != len(set(pairs)):
        raise SystemExit("Repeated-sampling plan contains duplicate cells.")

    cells = [load_cell(spec) for spec in specs]
    per_cell: dict[str, dict] = {}
    for cell in cells:
        per_cell[cell.label] = {
            "adapter": cell.model_family,
            "dataset": cell.dataset,
            "n": len(cell.y_error),
            "metrics": {
                metric: {
                    "trajuq": metric_value(
                        cell.y_error,
                        cell.candidate_score,
                        metric,
                    ),
                    "repeated_sampling": metric_value(
                        cell.y_error,
                        cell.baseline_score,
                        metric,
                    ),
                    "trajuq_improvement": oriented_improvement(
                        cell.y_error,
                        cell.candidate_score,
                        cell.baseline_score,
                        metric,
                    ),
                }
                for metric in AGGREGATE_METRICS
            },
        }

    macro = {
        metric: hierarchical_macro_bootstrap(
            cells,
            metric=metric,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        for metric in AGGREGATE_METRICS
    }

    payload = {
        "protocol": "trajuq_vs_repeated_macro_v1",
        "comparison": "single_trajectory_trajuq_vs_five_sample_baseline",
        "delta_direction": "positive_means_trajuq_is_better",
        "plan": args.plan,
        "expected_cells": expected,
        "observed_cells": len(cells),
        "cell_set_complete": len(cells) == expected,
        "eligible_datasets": plan.get("eligible_datasets"),
        "excluded_datasets": plan.get("excluded_datasets"),
        "per_cell": per_cell,
        "macro": macro,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
