#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

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
    parser.add_argument(
        "--primary-config",
        default="configs/experiments/primary.yaml",
        help="Frozen primary matrix used to reject selective cell omission.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exploratory only: allow aggregating fewer than the frozen primary cells.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    labels = [label for label, _ in args.cell]
    if len(labels) != len(set(labels)):
        raise SystemExit("Duplicate cell labels")

    cells = [load_cell(label, path) for label, path in args.cell]

    observed_pairs = [
        (cell.model_family, cell.dataset)
        for cell in cells
    ]
    if any(adapter is None or dataset is None for adapter, dataset in observed_pairs):
        raise SystemExit(
            "Prediction artifacts must include adapter and dataset metadata for aggregation."
        )
    if len(observed_pairs) != len(set(observed_pairs)):
        raise SystemExit("Duplicate model-family/dataset cells supplied to aggregation.")

    expected_pairs: set[tuple[str, str]] = set()
    primary_cfg = Path(args.primary_config)
    if primary_cfg.exists():
        cfg = yaml.safe_load(primary_cfg.read_text(encoding="utf-8"))
        adapters: list[str] = []
        for model_path in cfg.get("models", {}).get("primary", []):
            model_cfg = yaml.safe_load(Path(model_path).read_text(encoding="utf-8"))
            adapters.append(str(model_cfg["adapter"]))
        datasets = [str(name) for name in cfg.get("datasets", [])]
        expected_pairs = {(adapter, dataset) for adapter in adapters for dataset in datasets}
    elif not args.allow_partial:
        raise SystemExit(f"Missing primary config: {primary_cfg}")

    observed_set = set(observed_pairs)
    if expected_pairs and not args.allow_partial and observed_set != expected_pairs:
        missing = sorted(expected_pairs - observed_set)
        extra = sorted(observed_set - expected_pairs)
        raise SystemExit(
            "Primary aggregation cell set mismatch: "
            f"missing={missing} extra={extra}"
        )

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
        "primary_config": str(primary_cfg),
        "allow_partial": bool(args.allow_partial),
        "expected_cell_count": len(expected_pairs) if expected_pairs else None,
        "observed_cell_count": len(cells),
        "cell_set_complete": (
            observed_set == expected_pairs if expected_pairs else None
        ),
        "per_cell": per_cell,
        "macro": macro,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
