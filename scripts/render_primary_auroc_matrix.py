#!/usr/bin/env python3
"""Render the provenance-bound RQ1 cell matrix for the manuscript.

This is a presentation-only renderer over the authorized aggregate. It does
not recompute outcomes, fit models, or modify any scientific result.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


INK = "#202124"
RED = "#B64A4A"
TEAL = "#1F6F5F"
PAPER = "#F7F7F4"
PDF_METADATA = {
    "Creator": "TrajUQ artifact renderer",
    "Producer": "Matplotlib",
    "CreationDate": datetime(2026, 11, 3, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 11, 3, tzinfo=timezone.utc),
}


def render(aggregate_path: Path, config_path: Path, output_pdf: Path) -> None:
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if aggregate.get("cell_set_complete") is not True or aggregate.get("expected_cell_count") != 18:
        raise SystemExit("RQ1 aggregate is not the complete registered 18-cell matrix")

    configured = [Path(str(x)).stem for x in config["models"]["primary"]]
    adapters = [name for name in ("llada", "dream", "sdar") if name in configured]
    datasets = [str(x) for x in config["datasets"]]
    by_pair = {
        (cell["model_family"], cell["dataset"]): cell
        for cell in aggregate["per_cell"].values()
    }
    matrix = np.array(
        [[by_pair[(model, dataset)]["metrics"]["auroc"]["improvement"] for dataset in datasets]
         for model in adapters]
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white", "svg.hashsalt": "trajuq-rq1-matrix-v2",
    })
    dataset_labels = {
        "gsm8k": "GSM8K", "math500": "MATH-500", "gpqa_diamond": "GPQA-Diamond",
        "arc_challenge": "ARC-Challenge", "mmlu_pro": "MMLU-Pro", "truthfulqa": "TruthfulQA",
    }
    model_labels = {"llada": "LLaDA", "dream": "Dream", "sdar": "SDAR"}
    bound = max(float(np.max(np.abs(matrix))), 1e-6)
    cmap = LinearSegmentedColormap.from_list("rq1_div", [RED, PAPER, TEAL])
    fig, ax = plt.subplots(figsize=(7.15, 2.55))
    image = ax.imshow(
        matrix, cmap=cmap,
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto",
    )
    ax.set_xticks(range(len(datasets)), [dataset_labels[d] for d in datasets], rotation=18, ha="right")
    ax.set_yticks(range(len(adapters)), [model_labels[m] for m in adapters])
    ax.set_xlabel("Frozen benchmark task")
    ax.set_ylabel("Diffusion family")
    ax.set_xticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(adapters), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if abs(value) >= 0.12 else INK
            ax.text(column, row, f"{value:+.3f}", ha="center", va="center",
                    color=color, fontsize=8.0, fontweight=600)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    colorbar.set_label(r"$\Delta$AUROC vs. decision-only")
    fig.tight_layout(pad=0.25)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", metadata=PDF_METADATA)
    fig.savefig(output_pdf.with_suffix(".svg"), bbox_inches="tight",
                metadata={"Creator": "TrajUQ artifact renderer", "Date": "2026-11-03"})
    fig.savefig(output_pdf.with_suffix(".png"), bbox_inches="tight", dpi=300,
                metadata={"Software": "TrajUQ artifact renderer"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.aggregate, args.config, args.output)


if __name__ == "__main__":
    main()
