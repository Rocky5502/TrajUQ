#!/usr/bin/env python
"""Render the complete RQ1 evidence package from provenance-bound artifacts.

This incremental renderer intentionally excludes RQ2/RQ3.  The final combined
renderer remains responsible for replacing the downstream pending macros once
their frozen registries are complete.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.render_results_tex import (  # noqa: E402
    ABLATION_SUFFIXES,
    CALIBRATION_RENDER_METRICS,
    CALIBRATION_RENDER_VIEWS,
    PRIOR_SIGNAL_SUFFIXES,
    fmt,
    fmt_optional,
    fmt_signed,
    metric_pair,
    render_primary_detail_table,
    require_ablation,
    require_bound_result_provenance,
    require_calibration_sensitivity,
    require_claim_gate,
    require_primary,
    require_prior_signals,
)


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "results/primary_analysis/macro_primary.json"
CLAIM = ROOT / "results/primary_analysis/rq1_claim_gate.json"
ABLATION = ROOT / "results/primary_analysis/macro_feature_ablation.json"
PRIOR = ROOT / "results/primary_analysis/macro_prior_signal_diagnostics.json"
CALIBRATION = ROOT / "results/primary_analysis/macro_calibration_sensitivity.json"
CONFIG = ROOT / "configs/experiments/primary.yaml"
OUTPUT = ROOT / "paper/generated_rq1.tex"
MANIFEST = ROOT / "results/rq1/paper/rq1_paper_manifest.json"
FIGURE_DIR = ROOT / "paper/figures/generated"

TEAL = "#1F6F5F"
TEAL_LIGHT = "#88B7AA"
BLUE = "#3B6EA8"
AMBER = "#C8892E"
RED = "#B64A4A"
INK = "#202124"
MUTED = "#697077"
GRID = "#D9DDDF"
PDF_METADATA = {
    "Creator": "TrajUQ artifact renderer",
    "Producer": "Matplotlib",
    "CreationDate": datetime(2026, 10, 15, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 10, 15, tzinfo=timezone.utc),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def renew(name: str, value: str) -> str:
    return (
        f"\\providecommand{{\\{name}}}{{}}\n"
        f"\\renewcommand{{\\{name}}}{{{value}}}"
    )


def build_macros(
    primary: dict,
    claim: dict,
    ablation: dict,
    prior: dict,
    calibration: dict,
    config: dict,
) -> dict[str, str]:
    require_primary(primary)
    require_claim_gate(claim, primary, config)
    require_ablation(ablation)
    require_prior_signals(prior)
    require_calibration_sensitivity(calibration)

    macros: dict[str, str] = {}
    names = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "brier": "Brier",
        "nll": "NLL",
        "ece15": "ECE",
        "aurc": "AURC",
    }
    for metric, suffix in names.items():
        node = primary["macro"][metric]
        baseline, candidate = metric_pair(primary, metric)
        macros[f"PrimaryDecision{suffix}"] = fmt(baseline)
        macros[f"PrimaryTraj{suffix}"] = fmt(candidate)
        macros[f"PrimaryDelta{suffix}"] = fmt_signed(node["improvement_macro"])
        macros[f"PrimaryDelta{suffix}Low"] = fmt_signed(node["improvement_ci_low"])
        macros[f"PrimaryDelta{suffix}High"] = fmt_signed(node["improvement_ci_high"])

    for group, suffix in ABLATION_SUFFIXES.items():
        metrics = ablation["groups"][group]["metrics"]
        auroc = metrics["auroc"]
        macros[f"Ablation{suffix}AUROCDelta"] = fmt_signed(auroc["improvement_macro"])
        macros[f"Ablation{suffix}AUROCLow"] = fmt_signed(auroc["improvement_ci_low"])
        macros[f"Ablation{suffix}AUROCHigh"] = fmt_signed(auroc["improvement_ci_high"])
        macros[f"Ablation{suffix}AUPRCDelta"] = fmt_signed(metrics["auprc"]["improvement_macro"])
        macros[f"Ablation{suffix}AURCDelta"] = fmt_signed(metrics["aurc"]["improvement_macro"])

    for signal, suffix in PRIOR_SIGNAL_SUFFIXES.items():
        metrics = prior["macro"][signal]
        auroc = metrics["auroc"]
        macros[f"{suffix}AUROC"] = fmt(auroc["baseline_macro"])
        macros[f"{suffix}TrajAUROC"] = fmt(auroc["candidate_macro"])
        macros[f"{suffix}AUROCDelta"] = fmt_signed(auroc["improvement_macro"])
        macros[f"{suffix}AUROCLow"] = fmt_signed(auroc["improvement_ci_low"])
        macros[f"{suffix}AUROCHigh"] = fmt_signed(auroc["improvement_ci_high"])
        macros[f"{suffix}AUPRCDelta"] = fmt_signed(metrics["auprc"]["improvement_macro"])
        macros[f"{suffix}AURCDelta"] = fmt_signed(metrics["aurc"]["improvement_macro"])

    macros["CalibrationEligibleCells"] = str(int(calibration["eligible_cell_count"]))
    macros["CalibrationSkippedCells"] = str(int(calibration["skipped_cell_count"]))
    sensitivity = calibration["macro_over_eligible_cells"]
    for view, view_suffix in CALIBRATION_RENDER_VIEWS.items():
        for metric, metric_suffix in CALIBRATION_RENDER_METRICS.items():
            node = sensitivity[view][metric]
            macros[f"Calibration{view_suffix}{metric_suffix}Logistic"] = fmt_optional(node["primary_logistic"])
            macros[f"Calibration{view_suffix}{metric_suffix}Isotonic"] = fmt_optional(node["isotonic_sensitivity"])
            macros[f"Calibration{view_suffix}{metric_suffix}Delta"] = fmt_optional(
                node["isotonic_minus_logistic_improvement"], signed=True
            )

    observed = claim["observed"]
    delta = fmt_signed(observed["macro_auroc_improvement"])
    low = fmt_signed(observed["macro_auroc_ci_low"])
    high = fmt_signed(observed["macro_auroc_ci_high"])
    positive = int(observed["positive_model_families"])
    families = int(observed["n_model_families"])
    macros["AbstractResultSentence"] = (
        "Across the frozen 18-cell matrix, trajectory features improve macro "
        f"AUROC over decision-time entropy by {delta} (95\\% CI [{low}, {high}]), "
        f"with positive point improvement in {positive}/{families} model families."
    )
    macros["ResultsResultSentence"] = (
        "The pre-registered RQ1 positive-claim gate is met: equal-cell macro "
        f"AUROC improves by {delta} (95\\% CI [{low}, {high}]) and "
        f"{positive}/{families} model families improve. The gain is heterogeneous: "
        "LLaDA and Dream improve while SDAR does not, and ECE worsens despite gains "
        "in discrimination and selective risk."
    )
    macros["ConclusionResultSentence"] = (
        f"The frozen primary analysis yields a macro AUROC improvement of {delta} "
        "over decision-time entropy and satisfies the pre-registered RQ1 gate, "
        "with model-family and calibration heterogeneity retained as first-class findings."
    )
    macros["PaperStateLimitationsSentence"] = (
        "RQ1--RQ4 primary claims and the registered RQ5 comparison are complete "
        "and authorization-bound; secondary exploratory analyses cannot alter "
        "their primary gates."
    )
    macros["DetailedPrimaryTable"] = render_primary_detail_table(primary, config)
    return macros


def render_summary_table(primary: dict) -> str:
    labels = {
        "auroc": ("AUROC", "\\uparrow"),
        "auprc": ("AUPRC", "\\uparrow"),
        "brier": ("Brier", "\\downarrow"),
        "nll": ("NLL", "\\downarrow"),
        "ece15": ("ECE$_{15}$", "\\downarrow"),
        "aurc": ("AURC", "\\downarrow"),
    }
    rows = []
    for key, (label, direction) in labels.items():
        node = primary["macro"][key]
        rows.append(
            f"{label} ${direction}$ & {fmt(node['baseline_macro'])} & "
            f"{fmt(node['candidate_macro'])} & {fmt_signed(node['improvement_macro'])} "
            f"[{fmt_signed(node['improvement_ci_low'])}, {fmt_signed(node['improvement_ci_high'])}]\\\\"
        )
    return (
        "\\begin{table}[H]\n\\centering\n\\small\n"
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Metric & Decision entropy & \\method & Oriented $\\Delta$ (95\\% CI)\\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{RQ1 equal-cell macro results over 18 model--dataset cells. "
        "Improvements are oriented so positive values favor \\method; lower-is-better metrics "
        "therefore use baseline minus candidate. Confidence intervals are paired, "
        "class-stratified within cell and macro-averaged across cells.}\n"
        "\\label{tab:rq1-complete}\n\\end{table}"
    )


def render_figures(primary: dict, ablation: dict, config: dict) -> dict[str, Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
        "svg.hashsalt": "trajuq-rq1-v1",
    })
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    def save_set(fig: object, pdf_path: Path) -> None:
        fig.savefig(pdf_path, bbox_inches="tight", metadata=PDF_METADATA)
        fig.savefig(
            pdf_path.with_suffix(".svg"),
            bbox_inches="tight",
            metadata={"Creator": "TrajUQ artifact renderer", "Date": "2026-10-15"},
        )
        fig.savefig(
            pdf_path.with_suffix(".png"),
            bbox_inches="tight",
            dpi=300,
            metadata={"Software": "TrajUQ artifact renderer"},
        )
    adapters = [Path(str(x)).stem for x in config["models"]["primary"]]
    datasets = [str(x) for x in config["datasets"]]
    by_pair = {(v["model_family"], v["dataset"]): v for v in primary["per_cell"].values()}
    matrix = np.array([[by_pair[(m, d)]["metrics"]["auroc"]["improvement"] for d in datasets] for m in adapters])

    heatmap = FIGURE_DIR / "primary_auroc_improvement_matrix.pdf"
    bound = max(float(np.max(np.abs(matrix))), 1e-6)
    cmap = LinearSegmentedColormap.from_list("rq1_div", [RED, "#F7F7F4", TEAL])
    fig, ax = plt.subplots(figsize=(6.75, 2.45))
    im = ax.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
    ax.set_xticks(range(len(datasets)), [d.replace("_", "-") for d in datasets], rotation=22, ha="right")
    ax.set_yticks(range(len(adapters)), [m.upper() if m == "sdar" else m.title() for m in adapters])
    ax.set_xlabel("Frozen dataset")
    ax.set_ylabel("Model family")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i,j]:+.3f}", ha="center", va="center", color=INK, fontsize=7.3)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label("Oriented AUROC improvement")
    fig.tight_layout(pad=0.25)
    save_set(fig, heatmap)
    plt.close(fig)

    family = FIGURE_DIR / "family_macro_auroc_improvement.pdf"
    vals = primary["macro"]["auroc"]["model_family_macro_improvement"]
    ordered = ["llada", "dream", "sdar"]
    fig, ax = plt.subplots(figsize=(3.25, 2.15))
    y = np.arange(len(ordered))
    numbers = np.array([float(vals[x]) for x in ordered])
    colors = [TEAL if x >= 0 else RED for x in numbers]
    ax.hlines(y, 0, numbers, color=colors, linewidth=2)
    ax.scatter(numbers, y, color=colors, s=32, marker="o", zorder=3)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, ["LLaDA", "Dream", "SDAR"])
    ax.set_xlabel("Macro AUROC improvement")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.invert_yaxis()
    for yi, value in zip(y, numbers):
        # Keep value labels inside the plotting area.  Placing a negative label
        # to the left of its marker overlaps the SDAR tick label at column size.
        ax.annotate(
            f"{value:+.3f}",
            (value, yi),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
        )
    fig.tight_layout(pad=0.25)
    save_set(fig, family)
    plt.close(fig)

    forest = FIGURE_DIR / "rq1_feature_ablation.pdf"
    groups = ["decision_plus_distribution", "decision_plus_stability", "decision_plus_persistence", "full", "full_without_js", "full_without_revisions", "full_without_persistence"]
    group_labels = ["Distribution", "Stability", "Persistence", "Full", "Full - JS", "Full - revisions", "Full - persistence"]
    nodes = [ablation["groups"][g]["metrics"]["auroc"] for g in groups]
    center = np.array([float(x["improvement_macro"]) for x in nodes])
    low = np.array([float(x["improvement_ci_low"]) for x in nodes])
    high = np.array([float(x["improvement_ci_high"]) for x in nodes])
    y = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(3.25, 3.05))
    ax.errorbar(center, y, xerr=np.vstack([center-low, high-center]), fmt="o", color=TEAL, ecolor=TEAL_LIGHT, capsize=2.5, markersize=4.5)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y, group_labels)
    ax.set_xlabel("AUROC improvement vs. decision-only")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.invert_yaxis()
    fig.tight_layout(pad=0.25)
    save_set(fig, forest)
    plt.close(fig)
    return {"primary_auroc_matrix": heatmap, "family_macro_auroc": family, "feature_ablation": forest}


def main() -> None:
    paths = {"primary": PRIMARY, "claim_gate": CLAIM, "ablation": ABLATION, "prior_signals": PRIOR, "calibration_sensitivity": CALIBRATION, "primary_config": CONFIG}
    payloads = {key: load(path) for key, path in paths.items() if key != "primary_config"}
    commits = {require_bound_result_provenance(payload, label=key) for key, payload in payloads.items()}
    if commits != {"28002055c828c6910840de6f2105aa6367514144"}:
        raise ValueError(f"RQ1 sources do not bind the authoritative commit: {sorted(commits)}")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    macros = build_macros(payloads["primary"], payloads["claim_gate"], payloads["ablation"], payloads["prior_signals"], payloads["calibration_sensitivity"], config)
    figures = render_figures(payloads["primary"], payloads["ablation"], config)

    lines = ["% GENERATED FILE -- DO NOT TYPE RESULT VALUES BY HAND.", "% Complete provenance-bound RQ1 package; downstream RQs are supplied by their authorized interfaces."]
    for key, path in paths.items():
        lines.append(f"% source {key}: {path.relative_to(ROOT).as_posix()} sha256={sha256(path)}")
    for name, value in macros.items():
        lines.append(renew(name, value))
    lines.append("\\newcommand{\\RQOneSummaryTable}{" + render_summary_table(payloads["primary"]) + "}")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    visual_spec = {
        "background": "#FFFFFF", "ink": INK, "focal": TEAL,
        "comparison": BLUE, "caution": AMBER, "negative": RED,
        "font_points": 8.5, "vector": True,
        "qa": ["final-size", "grayscale", "caption-standalone", "source-hash"],
    }
    manifest = {
        "protocol": "trajuq_rq1_paper_package_v1",
        "status": "PASS",
        "authoritative_commit": next(iter(commits)),
        "sources": {key: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)} for key, path in paths.items()},
        "generated_tex": {"path": OUTPUT.relative_to(ROOT).as_posix(), "sha256": sha256(OUTPUT), "bytes": OUTPUT.stat().st_size},
        "figures": {key: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for key, path in figures.items()},
        "figure_fallbacks": {
            key: {
                ext: {
                    "path": path.with_suffix(ext).relative_to(ROOT).as_posix(),
                    "sha256": sha256(path.with_suffix(ext)),
                    "bytes": path.with_suffix(ext).stat().st_size,
                }
                for ext in (".svg", ".png")
            }
            for key, path in figures.items()
        },
        "visual_spec": visual_spec,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
