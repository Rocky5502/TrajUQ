#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from trajuq.aggregation import (
    PredictionCell,
    hierarchical_macro_bootstrap,
    metric_value,
    oriented_improvement,
)
from trajuq.io import read_jsonl

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.execute_repeated_sampling import validate_plan_provenance  # noqa: E402


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


def load_auxiliary_compute(spec: dict, expected_n: int) -> dict[str, float | int | str]:
    label = str(spec["cell"])
    metrics_path = Path(str(spec["repeated_metrics"]))
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label}: cannot read repeated metrics for compute accounting: {exc}") from exc

    if payload.get("protocol") != "trajuq_repeated_sampling_baseline_v2":
        raise SystemExit(f"{label}: repeated metrics protocol is not v2")
    if int(payload.get("n_test", -1)) != int(expected_n):
        raise SystemExit(f"{label}: repeated metrics test count is incomplete")

    expected_samples = len([int(seed) for seed in spec.get("seeds", [])])
    if expected_samples <= 0:
        raise SystemExit(f"{label}: repeated plan has no frozen sample seeds")
    if int(payload.get("sample_count", -1)) != expected_samples:
        raise SystemExit(
            f"{label}: repeated sample count drifted: "
            f"{payload.get('sample_count')!r} != {expected_samples}"
        )

    test_cost = payload.get("auxiliary_compute", {}).get("test", {})
    values: dict[str, float] = {}
    for key in (
        "mean_auxiliary_wall_time_seconds_per_example",
        "p90_auxiliary_wall_time_seconds_per_example",
    ):
        value = test_cost.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"{label}: missing/non-numeric repeated test compute field {key}"
            ) from exc
        if not np.isfinite(number) or number < 0:
            raise SystemExit(
                f"{label}: invalid repeated test compute field {key}={number!r}"
            )
        values[key] = number

    return {
        "sample_count": expected_samples,
        **values,
        "definition": (
            "extra repeated-generation wall time for the frozen auxiliary samples; "
            "target generation excluded"
        ),
    }


def load_cell(spec: dict) -> PredictionCell:
    label = str(spec["cell"])
    traj = index(str(spec["trajuq_predictions"]))
    repeated = index(str(spec["repeated_predictions"]))
    expected = index(
        str(spec["provenance"]["test"]["scored_target"])
    )
    if set(traj) != set(repeated):
        raise SystemExit(f"{label}: prediction ID sets differ")
    if set(traj) != set(expected):
        raise SystemExit(
            f"{label}: repeated comparison does not exactly cover "
            "the frozen scored test target"
        )

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
    provenance_failures = validate_plan_provenance(plan)
    if provenance_failures:
        raise SystemExit(
            "Repeated-sampling plan provenance check failed:\n- "
            + "\n- ".join(provenance_failures)
        )
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
    compute_rows: list[dict[str, float | int | str]] = []
    for spec, cell in zip(specs, cells):
        compute = load_auxiliary_compute(spec, len(cell.y_error))
        compute_rows.append(compute)
        per_cell[cell.label] = {
            "adapter": cell.model_family,
            "dataset": cell.dataset,
            "n": len(cell.y_error),
            "auxiliary_compute": compute,
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

    mean_costs = np.asarray(
        [
            float(row["mean_auxiliary_wall_time_seconds_per_example"])
            for row in compute_rows
        ],
        dtype=float,
    )
    p90_costs = np.asarray(
        [
            float(row["p90_auxiliary_wall_time_seconds_per_example"])
            for row in compute_rows
        ],
        dtype=float,
    )
    sample_counts = {int(row["sample_count"]) for row in compute_rows}
    if len(sample_counts) != 1:
        raise SystemExit(
            f"Repeated-sampling cells mix sample counts: {sorted(sample_counts)}"
        )

    auxiliary_compute = {
        "definition": (
            "extra five-sample repeated-generation wall time; "
            "target generation excluded"
        ),
        "aggregation": "equal_cell_weight_macro",
        "expected_cells": expected,
        "observed_cells_with_complete_test_wall_time": len(compute_rows),
        "complete": len(compute_rows) == expected,
        "sample_count": next(iter(sample_counts)),
        "mean_test_auxiliary_wall_time_seconds_per_example": float(
            mean_costs.mean()
        ),
        "mean_test_p90_auxiliary_wall_time_seconds_per_example": float(
            p90_costs.mean()
        ),
    }

    payload = {
        "protocol": "trajuq_vs_repeated_macro_v2",
        "comparison": "single_trajectory_trajuq_vs_five_sample_baseline",
        "delta_direction": "positive_means_trajuq_is_better",
        "plan": args.plan,
        "expected_cells": expected,
        "observed_cells": len(cells),
        "cell_set_complete": len(cells) == expected,
        "eligible_datasets": plan.get("eligible_datasets"),
        "excluded_datasets": plan.get("excluded_datasets"),
        "auxiliary_compute": auxiliary_compute,
        "per_cell": per_cell,
        "macro": macro,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
