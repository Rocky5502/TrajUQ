#!/usr/bin/env python3
"""Render evidence-locked RQ2 and RQ3 main-paper figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INK = "#171717"
SECONDARY = "#5E6462"
TEAL = "#1F6F5F"
LIGHT_TEAL = "#8EAAA2"
BASELINE = "#8D8D88"
LIGHT = "#D8D8D4"
GRID = "#E8E8E5"
RISK = "#B5473C"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.edgecolor": LIGHT,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": SECONDARY,
            "ytick.color": SECONDARY,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "trajuq-rq2-rq3-main-v1",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_three(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs = [base.with_suffix(suffix) for suffix in (".pdf", ".svg", ".png")]
    fig.savefig(outputs[0], metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(outputs[1], metadata={"Date": None})
    fig.savefig(outputs[2], dpi=300, metadata={"Date": None})
    return outputs


def validate_sources(evidence_root: Path, spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for relative, expected_hash in spec["source_hashes"].items():
        path = evidence_root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"evidence drift: {relative}")
        payloads[relative] = load_json(path)
    rq2 = payloads["results/repeated_sampling/macro_trajuq_vs_repeated.json"]
    if (
        rq2.get("protocol") != "trajuq_vs_repeated_macro_v2"
        or not rq2.get("cell_set_complete")
        or rq2.get("observed_cells") != 12
    ):
        raise ValueError("RQ2 evidence is not the authorized complete 12-cell macro")
    for key in ("025", "050", "075"):
        adaptive = payloads[f"results/adaptive/macro_cap_{key}.json"]
        if (
            adaptive.get("protocol") != "trajuq_adaptive_macro_v1"
            or not adaptive.get("cell_set_complete")
            or adaptive.get("observed_cells") != 18
        ):
            raise ValueError(f"RQ3 cap {key} is incomplete")
    return payloads


def render_rq2(rq2: dict[str, Any], output_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    metrics = ["aurc", "auprc", "auroc", "ece15", "nll"]
    labels = ["AURC ↓", "AUPRC ↑", "AUROC ↑", "ECE₁₅ ↓", "NLL ↓"]
    baseline = np.array([rq2["macro"][metric]["baseline_macro"] for metric in metrics])
    candidate = np.array([rq2["macro"][metric]["candidate_macro"] for metric in metrics])
    improvement = np.array([rq2["macro"][metric]["improvement_macro"] for metric in metrics])
    ci_low = np.array([rq2["macro"][metric]["improvement_ci_low"] for metric in metrics])
    ci_high = np.array([rq2["macro"][metric]["improvement_ci_high"] for metric in metrics])

    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(178 / 25.4, 86 / 25.4),
        gridspec_kw={"width_ratios": [1.08, 0.92], "wspace": 0.43},
    )
    y = np.arange(len(metrics))
    for index in range(len(metrics)):
        left.plot([baseline[index], candidate[index]], [index, index], color=LIGHT, linewidth=1.2)
    left.scatter(baseline, y, color=BASELINE, marker="o", s=23, label="Five samples", zorder=3)
    left.scatter(candidate, y, color=TEAL, marker="D", s=23, label="TrajUQ", zorder=3)
    left.set_yticks(y, labels)
    left.invert_yaxis()
    left.set_xlabel("Macro metric value")
    left.set_title("(a) Macro values and uncertainty", loc="left", fontweight="semibold", fontsize=9.0)
    left.grid(axis="x", color=GRID, linewidth=0.6)
    left.set_axisbelow(True)
    left.legend(frameon=False, ncol=1, loc="upper left")
    for index, (delta, low, high) in enumerate(zip(improvement, ci_low, ci_high, strict=True)):
        color = TEAL if low > 0 else (RISK if high < 0 else SECONDARY)
        left.text(
            max(baseline[index], candidate[index]) + 0.018,
            index,
            f"Δ {delta:+.3f} [{low:+.3f}, {high:+.3f}]",
            va="center",
            fontsize=8.0,
            color=color,
        )
    left.set_xlim(0.0, max(float(max(baseline.max(), candidate.max())) + 0.19, 0.83))

    families = ["dream", "llada", "sdar"]
    family_labels = ["Dream", "LLaDA", "SDAR"]
    family_y = np.arange(len(families))
    right.axvline(0.0, color=INK, linewidth=0.8)
    for offset, metric, marker, name in [(-0.10, "aurc", "o", "AURC improvement"), (0.10, "auprc", "s", "AUPRC improvement")]:
        values = [rq2["macro"][metric]["model_family_macro_improvement"][family] for family in families]
        right.scatter(values, family_y + offset, marker=marker, s=25, color=TEAL if metric == "aurc" else SECONDARY, label=name, zorder=3)
    right.set_yticks(family_y, family_labels)
    right.invert_yaxis()
    right.set_xlabel("Improvement over five samples")
    right.set_title("(b) Family heterogeneity", loc="left", fontweight="semibold", fontsize=9.0)
    right.grid(axis="x", color=GRID, linewidth=0.6)
    right.set_axisbelow(True)
    right.legend(frameon=False, loc="upper right")
    right.set_xlim(-0.04, 0.075)

    fig.suptitle(
        "Trajectory risk improves selective ranking on average, but not uniformly",
        x=0.075,
        y=0.97,
        ha="left",
        fontsize=10.5,
        fontweight="semibold",
    )
    fig.text(
        0.075,
        0.025,
        "CIs: registered hierarchical 95% bootstrap intervals (12 cells).\nFamily dots have no fabricated intervals.",
        fontsize=8.0,
        color=SECONDARY,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.79, bottom=0.22)
    outputs = save_three(fig, output_dir / "rq2_selective_vs_repeated")
    plt.close(fig)
    return outputs, {
        "metrics": metrics,
        "baseline": baseline.tolist(),
        "candidate": candidate.tolist(),
        "improvement": improvement.tolist(),
        "ci_low": ci_low.tolist(),
        "ci_high": ci_high.tolist(),
    }


def render_rq3(adaptive: dict[str, dict[str, Any]], output_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    keys = ["025", "050", "075"]
    costs = np.array([adaptive[key]["macro"]["nominal_cost_ratio_to_base"] for key in keys])
    accuracy = np.array([adaptive[key]["macro"]["adaptive_accuracy"] for key in keys])
    x_low = np.array([adaptive[key]["intervals"]["nominal_cost_ratio_to_base"]["ci_low"] for key in keys])
    x_high = np.array([adaptive[key]["intervals"]["nominal_cost_ratio_to_base"]["ci_high"] for key in keys])
    y_low = np.array([adaptive[key]["intervals"]["adaptive_accuracy"]["ci_low"] for key in keys])
    y_high = np.array([adaptive[key]["intervals"]["adaptive_accuracy"]["ci_high"] for key in keys])
    random_accuracy = np.array([adaptive[key]["macro"]["random_expected_accuracy"] for key in keys])
    base_accuracy = float(adaptive["050"]["macro"]["base_accuracy"])
    refined_accuracy = float(adaptive["050"]["macro"]["refined_accuracy"])
    refined_cost = float(adaptive["050"]["macro"]["refined_cost_ratio_to_base"])

    fig, ax = plt.subplots(figsize=(85 / 25.4, 62 / 25.4))
    ax.plot(costs, accuracy, color=LIGHT_TEAL, linewidth=1.2, zorder=1)
    for index, key in enumerate(keys):
        color = TEAL if key == "050" else LIGHT_TEAL
        marker = "D" if key == "050" else "o"
        ax.errorbar(
            costs[index],
            accuracy[index],
            xerr=[[costs[index] - x_low[index]], [x_high[index] - costs[index]]],
            yerr=[[accuracy[index] - y_low[index]], [y_high[index] - accuracy[index]]],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            markersize=5.0,
            zorder=4,
        )
    ax.scatter(costs, random_accuracy, color=BASELINE, marker="x", s=25, zorder=3)
    ax.scatter([1.0], [base_accuracy], color=INK, marker="o", s=24, zorder=3)
    ax.scatter([refined_cost], [refined_accuracy], facecolors="white", edgecolors=INK, marker="o", s=28, zorder=3)
    ax.text(1.16, random_accuracy[0] + 0.0003, "Matched random", color=SECONDARY, fontsize=8.0, va="center")
    ax.text(1.05, base_accuracy - 0.0010, "Base", color=INK, fontsize=8.0, va="center")
    ax.text(refined_cost - 0.06, refined_accuracy + 0.0002, "Full", color=INK, fontsize=8.0, ha="right", va="center")
    ax.annotate("Primary 50% cap", xy=(costs[1], accuracy[1]), xytext=(2.04, 0.4055), color=TEAL, fontsize=8.0, ha="center", arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 0.7})
    ax.set_xlabel("Nominal compute / base · lower is better", fontsize=8.0)
    ax.set_ylabel("Accuracy · higher is better")
    ax.set_title("Risk guidance improves\nthe accuracy–compute trade-off", loc="left", fontweight="semibold", fontsize=9.0)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0.88, 3.50)
    ax.set_ylim(0.370, 0.410)
    fig.text(0.18, 0.055, "Registered hierarchical 95% bootstrap CIs.\nNear-full accuracy is not an equivalence claim.", fontsize=8.0, color=SECONDARY, va="bottom")
    fig.subplots_adjust(left=0.18, right=0.98, top=0.76, bottom=0.32)
    outputs = save_three(fig, output_dir / "rq3_accuracy_compute_tradeoff")
    plt.close(fig)
    return outputs, {
        "keys": keys,
        "costs": costs.tolist(),
        "accuracy": accuracy.tolist(),
        "random_accuracy": random_accuracy.tolist(),
        "base": [1.0, base_accuracy],
        "full_refinement": [refined_cost, refined_accuracy],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("configs/figures/rq2_rq3_main_figures.json"))
    args = parser.parse_args()
    evidence_root = args.evidence_root.resolve()
    spec_path = args.spec.resolve()
    spec = load_json(spec_path)
    payloads = validate_sources(evidence_root, spec)
    apply_style()
    rq2_outputs, rq2_values = render_rq2(payloads["results/repeated_sampling/macro_trajuq_vs_repeated.json"], args.output_dir.resolve())
    adaptive = {key: payloads[f"results/adaptive/macro_cap_{key}.json"] for key in ("025", "050", "075")}
    rq3_outputs, rq3_values = render_rq3(adaptive, args.output_dir.resolve())
    manifest = {
        "schema": "trajuq_rq2_rq3_main_figure_manifest_v1",
        "status": "PASS",
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "sources": spec["source_hashes"],
        "plotted_values": {"rq2": rq2_values, "rq3": rq3_values},
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in [*rq2_outputs, *rq3_outputs]
        ],
        "qa": {
            "final_size_mm": {"rq2": [178, 86], "rq3": [85, 62]},
            "minimum_text_pt": 8.0,
            "vector_outputs": True,
            "no_fabricated_intervals": True,
            "negative_and_null_evidence_visible": True,
            "grayscale_redundancy": True
        }
    }
    manifest_path = args.output_dir.resolve() / "rq2_rq3_main_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
