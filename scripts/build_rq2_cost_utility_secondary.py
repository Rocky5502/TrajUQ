#!/usr/bin/env python
"""Build the frozen RQ2 K=2/3/5 cost--utility secondary package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


EXPECTED_SOURCE_SHA256 = "bff7b1d84d8d22ecb8449cda3c58cce2adde76d20c6b9a75f7fe703a9b020247"
K_VALUES = (2, 3, 5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def incremental_rows(rows: list[dict]) -> list[dict]:
    result = []
    for earlier, later in zip(rows, rows[1:]):
        extra_generations = later["k"] - earlier["k"]
        extra_seconds = later["seconds"] - earlier["seconds"]
        aurc_reduction = earlier["aurc"] - later["aurc"]
        auroc_gain = later["auroc"] - earlier["auroc"]
        auprc_gain = later["auprc"] - earlier["auprc"]
        result.append(
            {
                "transition": f"K={earlier['k']} to K={later['k']}",
                "extra_generations": extra_generations,
                "extra_seconds_per_example": extra_seconds,
                "aurc_reduction": aurc_reduction,
                "aurc_reduction_per_extra_generation": aurc_reduction / extra_generations,
                "aurc_reduction_per_extra_second": aurc_reduction / extra_seconds,
                "auroc_gain": auroc_gain,
                "auroc_gain_per_extra_generation": auroc_gain / extra_generations,
                "auroc_gain_per_extra_second": auroc_gain / extra_seconds,
                "auprc_gain": auprc_gain,
                "auprc_gain_per_extra_generation": auprc_gain / extra_generations,
                "auprc_gain_per_extra_second": auprc_gain / extra_seconds,
            }
        )
    return result


def nondominated(rows: list[dict], utility: str, lower_is_better: bool) -> list[int]:
    frontier = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            cost_no_worse = other["seconds"] <= candidate["seconds"]
            utility_no_worse = (
                other[utility] <= candidate[utility]
                if lower_is_better
                else other[utility] >= candidate[utility]
            )
            strictly_better = other["seconds"] < candidate["seconds"] or other[utility] != candidate[utility]
            if cost_no_worse and utility_no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate["k"])
    return frontier


def render_figure(rows: list[dict], output: Path) -> list[dict]:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    seconds = [row["seconds"] for row in rows]
    colors = {"aurc": "#0072B2", "auroc": "#009E73", "auprc": "#CC79A7", "ece15": "#D55E00"}
    fig, axes = plt.subplots(1, 3, figsize=(7.01, 2.42), constrained_layout=True)

    axes[0].plot(seconds, [row["aurc"] for row in rows], "o-", color=colors["aurc"], linewidth=1.5)
    axes[0].set_ylabel("AURC (lower is better)")
    axes[0].set_title("Selective risk")

    axes[1].plot(seconds, [row["auroc"] for row in rows], "o-", color=colors["auroc"], linewidth=1.5, label="AUROC")
    axes[1].plot(seconds, [row["auprc"] for row in rows], "s--", color=colors["auprc"], linewidth=1.5, label="AUPRC")
    axes[1].set_ylabel("Ranking metric (higher is better)")
    axes[1].set_title("Error ranking")
    axes[1].legend(frameon=False, loc="center right")

    axes[2].plot(seconds, [row["ece15"] for row in rows], "o-", color=colors["ece15"], linewidth=1.5)
    axes[2].set_ylabel("ECE$_{15}$ (lower is better)")
    axes[2].set_title("Calibration trade-off")

    for panel, axis in zip("ABC", axes):
        axis.set_xlabel("Auxiliary seconds / example")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        axis.text(-0.18, 1.06, panel, transform=axis.transAxes, fontweight="bold", va="top")
        for row in rows:
            metric = "aurc" if panel == "A" else ("auroc" if panel == "B" else "ece15")
            axis.annotate(f"K={row['k']}", (row["seconds"], row[metric]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)

    artifacts = []
    for suffix in ("pdf", "svg", "png"):
        path = output / f"rq2_cost_utility.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        artifacts.append({"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    plt.close(fig)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite output: {output}")
    if sha256(source) != EXPECTED_SOURCE_SHA256:
        raise SystemExit("RQ2 sample-efficiency source drift")
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("status") != "PASS" or document.get("new_model_generations") != 0:
        raise SystemExit("RQ2 sample-efficiency source is not an authorized zero-generation PASS artifact")
    output.mkdir(parents=True)

    rows = []
    for k in K_VALUES:
        item = document["macro"][str(k)]
        metrics = item["metrics"]
        rows.append(
            {
                "k": k,
                "seconds": item["mean_auxiliary_wall_time_seconds_per_example"],
                "cost_fraction_of_k5": item["cost_fraction_of_k5"],
                "n_cells": item["n_cells"],
                "aurc": metrics["aurc"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "risk_at_50_coverage": metrics["risk_at_50_coverage"],
                "risk_at_80_coverage": metrics["risk_at_80_coverage"],
                "ece15": metrics["ece15"],
            }
        )
    increments = incremental_rows(rows)
    frontiers = {
        metric: nondominated(rows, metric, lower_is_better=metric in {"aurc", "ece15"})
        for metric in ("aurc", "auroc", "auprc", "ece15")
    }
    payload = {
        "schema": "trajuq_rq2_cost_utility_secondary_v1",
        "status": "PASS",
        "classification": "SECONDARY_EXPLORATORY",
        "scientific_claim_authority": False,
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "aggregation": document["aggregation"],
        "new_model_generations": 0,
        "macro": rows,
        "incremental_cost_utility": increments,
        "nondominated_k_by_metric": frontiers,
        "interpretation": {
            "ranking_and_selective_risk": "improve monotonically from K=2 to K=5 at increasing auxiliary cost",
            "calibration": "ECE15 worsens monotonically over the same K values",
            "boundary": "descriptive three-point sensitivity; not a continuously optimized stopping policy",
        },
    }
    write_json_new(output / "rq2_cost_utility.json", payload)
    for name, content in (("macro_metrics.csv", rows), ("incremental_cost_utility.csv", increments)):
        with (output / name).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(content[0]))
            writer.writeheader()
            writer.writerows(content)
    figure_artifacts = render_figure(rows, output)
    visual_spec = {
        "schema": "trajuq_visual_spec_v1",
        "figure": "rq2_cost_utility",
        "target_width_mm": 178,
        "minimum_text_pt": 8,
        "panels": ["selective risk", "error ranking", "calibration trade-off"],
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "accessibility": "colorblind-safe palette plus distinct markers and line styles",
    }
    write_json_new(output / "rq2_cost_utility.visual_spec.json", visual_spec)
    artifacts = []
    for path in sorted(output.iterdir()):
        if path.is_file():
            artifacts.append({"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest = {
        "schema": "trajuq_rq2_cost_utility_manifest_v1",
        "status": "PASS",
        "classification": "SECONDARY_EXPLORATORY",
        "scientific_claim_authority": False,
        "source_path": str(source),
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "artifacts": artifacts,
        "figure_artifacts": figure_artifacts,
    }
    write_json_new(output / "manifest.json", manifest)
    (output / "manifest.json.sha256").write_text(sha256(output / "manifest.json") + "\n", encoding="ascii", newline="\n")
    print(json.dumps({"status": "PASS", "output": str(output), "manifest_sha256": sha256(output / "manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
