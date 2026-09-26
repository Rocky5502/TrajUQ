#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing result artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON artifact {path}: {exc}") from exc


def primary_orders(config_path: Path) -> tuple[list[str], list[str]]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    adapters = [
        Path(str(model_path)).stem
        for model_path in cfg["models"]["primary"]
    ]
    if not adapters or len(adapters) != len(set(adapters)):
        raise ValueError("Primary model-config path stems are empty or duplicated.")
    datasets = [str(name) for name in cfg["datasets"]]
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("Primary dataset order is empty or duplicated.")
    return adapters, datasets


def auroc_improvement_matrix(
    primary: dict,
    adapters: list[str],
    datasets: list[str],
) -> np.ndarray:
    matrix = np.full((len(adapters), len(datasets)), np.nan, dtype=float)
    by_pair = {
        (str(cell["model_family"]), str(cell["dataset"])): cell
        for cell in primary["per_cell"].values()
    }
    for row, adapter in enumerate(adapters):
        for col, dataset in enumerate(datasets):
            key = (adapter, dataset)
            if key not in by_pair:
                raise ValueError(f"Missing primary result cell: {key}")
            matrix[row, col] = float(
                by_pair[key]["metrics"]["auroc"]["improvement"]
            )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Non-finite AUROC improvement in primary matrix.")
    return matrix


def require_complete_primary(primary: dict, expected: int) -> None:
    if primary.get("protocol") != "hierarchical_within_cell_macro_bootstrap_v1":
        raise ValueError("Primary result protocol is not recognized.")
    if primary.get("cell_set_complete") is not True:
        raise ValueError("Primary result cell set is incomplete.")
    if int(primary.get("expected_cell_count", -1)) != expected:
        raise ValueError("Primary result expected cell count is unexpected.")
    if int(primary.get("observed_cell_count", -1)) != expected:
        raise ValueError("Primary result observed cell count is unexpected.")


def require_complete_adaptive(
    payload: dict,
    expected: int,
    *,
    fraction_key: str,
    expected_fraction: float,
    is_primary: bool,
) -> None:
    if payload.get("protocol") != "trajuq_adaptive_macro_v1":
        raise ValueError("Adaptive result protocol is not recognized.")
    if payload.get("execution_semantics") != "selective_second_pass_rerun":
        raise ValueError("Adaptive result execution semantics drifted.")
    if payload.get("fraction_key") != fraction_key:
        raise ValueError("Adaptive result fraction key drifted.")
    if payload.get("cell_set_complete") is not True:
        raise ValueError("Adaptive result cell set is incomplete.")
    if int(payload.get("expected_cells", -1)) != expected:
        raise ValueError("Adaptive result expected cell count is unexpected.")
    if int(payload.get("observed_cells", -1)) != expected:
        raise ValueError("Adaptive result observed cell count is unexpected.")
    if bool(payload.get("is_primary_fraction")) is not is_primary:
        raise ValueError("Adaptive result primary-cap flag drifted.")
    if not np.isclose(
        float(payload.get("max_refine_fraction", -1)),
        expected_fraction,
    ):
        raise ValueError("Adaptive result refinement fraction drifted.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render paper-level vector result figures only from complete frozen "
            "TrajUQ aggregate artifacts."
        )
    )
    parser.add_argument(
        "--primary",
        default="results/primary_analysis/macro_primary.json",
    )
    parser.add_argument(
        "--adaptive-025",
        default="results/adaptive/macro_cap_025.json",
    )
    parser.add_argument(
        "--adaptive-050",
        default="results/adaptive/macro_cap_050.json",
    )
    parser.add_argument(
        "--adaptive-075",
        default="results/adaptive/macro_cap_075.json",
    )
    parser.add_argument(
        "--primary-config",
        default="configs/experiments/primary.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/figures/generated",
    )
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            'Install analysis extras first: pip install -e ".[analysis]"'
        ) from exc

    primary_path = Path(args.primary)
    adaptive_paths = {
        "cap_025": Path(args.adaptive_025),
        "cap_050": Path(args.adaptive_050),
        "cap_075": Path(args.adaptive_075),
    }
    primary = load_json(primary_path)
    adaptive = {
        key: load_json(path)
        for key, path in adaptive_paths.items()
    }

    adapters, datasets = primary_orders(Path(args.primary_config))
    expected = len(adapters) * len(datasets)
    require_complete_primary(primary, expected)
    expected_fractions = {
        "cap_025": (0.25, False),
        "cap_050": (0.50, True),
        "cap_075": (0.75, False),
    }
    for key, payload in adaptive.items():
        fraction, is_primary = expected_fractions[key]
        require_complete_adaptive(
            payload,
            expected,
            fraction_key=key,
            expected_fraction=fraction,
            is_primary=is_primary,
        )

    matrix = auroc_improvement_matrix(primary, adapters, datasets)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = out_dir / "primary_auroc_improvement_matrix.pdf"
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(adapters)))
    ax.set_yticklabels(adapters)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Diffusion model family")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                f"{matrix[row, col]:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="AUROC improvement over decision entropy")
    fig.tight_layout()
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)

    family_path = out_dir / "family_macro_auroc_improvement.pdf"
    family_values = primary["macro"]["auroc"].get(
        "model_family_macro_improvement",
        {},
    )
    if set(family_values) != set(adapters):
        raise SystemExit(
            "Primary macro does not contain exactly the frozen model families."
        )
    values = np.asarray(
        [float(family_values[adapter]) for adapter in adapters],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(4.7, 3.2))
    positions = np.arange(len(adapters))
    ax.bar(positions, values)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(adapters)
    ax.set_ylabel("Macro AUROC improvement")
    ax.set_xlabel("Model family")
    fig.tight_layout()
    fig.savefig(family_path, bbox_inches="tight")
    plt.close(fig)

    tradeoff_path = out_dir / "adaptive_accuracy_cost_tradeoff.pdf"
    ordered_keys = ["cap_025", "cap_050", "cap_075"]
    costs = np.asarray(
        [
            float(adaptive[key]["macro"]["nominal_cost_ratio_to_base"])
            for key in ordered_keys
        ],
        dtype=float,
    )
    accuracy = np.asarray(
        [
            float(adaptive[key]["macro"]["adaptive_accuracy"])
            for key in ordered_keys
        ],
        dtype=float,
    )
    caps = np.asarray(
        [float(adaptive[key]["max_refine_fraction"]) for key in ordered_keys],
        dtype=float,
    )
    base_accuracy = float(adaptive["cap_050"]["macro"]["base_accuracy"])
    refined_accuracy = float(
        adaptive["cap_050"]["macro"]["refined_accuracy"]
    )
    refined_cost = float(
        adaptive["cap_050"]["macro"]["refined_cost_ratio_to_base"]
    )

    fig, ax = plt.subplots(figsize=(4.9, 3.5))
    ax.plot(costs, accuracy, marker="o", label="TrajUQ adaptive")
    ax.scatter([1.0], [base_accuracy], marker="s", label="Base budget")
    ax.scatter(
        [refined_cost],
        [refined_accuracy],
        marker="^",
        label="Fixed refined",
    )
    for x, y, cap in zip(costs, accuracy, caps):
        ax.annotate(
            f"{int(round(cap * 100))}%",
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Nominal cost / base-generation cost")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(tradeoff_path, bbox_inches="tight")
    plt.close(fig)

    primary_config_path = Path(args.primary_config)
    sources = {
        "primary": {
            "path": str(primary_path),
            "sha256": sha256_file(primary_path),
        },
        "primary_config": {
            "path": str(primary_config_path),
            "sha256": sha256_file(primary_config_path),
        },
        **{
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in adaptive_paths.items()
        },
    }
    output_paths = {
        "primary_auroc_matrix": heatmap_path,
        "family_macro_auroc": family_path,
        "adaptive_tradeoff": tradeoff_path,
    }
    manifest = {
        "protocol": "trajuq_paper_result_figures_v2",
        "sources": sources,
        "expected_cells": expected,
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in output_paths.items()
        },
    }
    manifest_path = out_dir / "result_figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
