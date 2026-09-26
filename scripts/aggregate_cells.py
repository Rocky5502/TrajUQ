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


def parse_cell(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--cell must use LABEL=PATH, e.g. llada_gsm8k=results/predictions.jsonl"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("--cell label and path must be non-empty")
    return label, path


def load_cell(label: str, path: str) -> PredictionCell:
    rows = list(read_jsonl(path))
    if not rows:
        raise SystemExit(f"{label}: empty prediction file {path}")
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{label}: duplicate example IDs")

    y = np.asarray([int(row["y_error"]) for row in rows], dtype=int)

    adapters = {str(row.get("adapter")) for row in rows if row.get("adapter") is not None}
    datasets = {str(row.get("dataset")) for row in rows if row.get("dataset") is not None}
    if len(adapters) > 1:
        raise SystemExit(f"{label}: prediction file mixes adapters: {sorted(adapters)}")
    if len(datasets) > 1:
        raise SystemExit(f"{label}: prediction file mixes datasets: {sorted(datasets)}")
    model_family = next(iter(adapters)) if adapters else None
    dataset = next(iter(datasets)) if datasets else None

    baseline = np.asarray(
        [float(row["p_error_decision_entropy"]) for row in rows],
        dtype=float,
    )
    candidate = np.asarray(
        [float(row["p_error_trajuq"]) for row in rows],
        dtype=float,
    )
    return PredictionCell(
        label=label,
        y_error=y,
        candidate_score=candidate,
        baseline_score=baseline,
        model_family=model_family,
        dataset=dataset,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Macro-aggregate held-out TrajUQ prediction cells without pooling "
            "heterogeneous benchmark examples."
        )
    )
    parser.add_argument(
        "--cell",
        action="append",
        type=parse_cell,
        required=True,
        help="Repeat LABEL=PATH for each model-dataset prediction artifact.",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    labels = [label for label, _ in args.cell]
    if len(labels) != len(set(labels)):
        raise SystemExit("Duplicate cell labels")

    cells = [load_cell(label, path) for label, path in args.cell]
    per_cell: dict[str, dict] = {}
    for cell in cells:
        per_cell[cell.label] = {
            "model_family": cell.model_family,
            "dataset": cell.dataset,
            "n": len(cell.y_error),
            "error_rate": float(np.mean(cell.y_error)),
            "metrics": {
                metric: {
                    "candidate": metric_value(
                        cell.y_error,
                        cell.candidate_score,
                        metric,
                    ),
                    "baseline": metric_value(
                        cell.y_error,
                        cell.baseline_score,
                        metric,
                    ),
                    "improvement": oriented_improvement(
                        cell.y_error,
                        cell.candidate_score,
                        cell.baseline_score,
                        metric,
                    ),
                }
                for metric in ("auroc", "auprc", "aurc")
            },
        }

    macro = {
        metric: hierarchical_macro_bootstrap(
            cells,
            metric=metric,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        for metric in ("auroc", "auprc", "aurc")
    }

    payload = {
        "protocol": "hierarchical_within_cell_macro_bootstrap_v1",
        "delta_direction": "positive_means_trajuq_is_better",
        "cell_weighting": "equal_weight_per_model_dataset_cell",
        "resampling": "class_stratified_within_each_cell_paired",
        "seed": args.seed,
        "per_cell": per_cell,
        "macro": macro,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
