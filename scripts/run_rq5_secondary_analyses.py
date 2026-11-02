#!/usr/bin/env python
"""Existing-data-only RQ5 secondary decomposition and robustness analyses.

This script is deliberately downstream of the immutable confirmatory gate. It
verifies the exact registered result before producing any exploratory output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PRIMARY = {
    "estimate": -0.002990487543719062,
    "ci_95": [-0.015057308574727653, 0.010202548928115571],
    "holm_adjusted_p": 0.6877312268773123,
    "decision": "NOT_SUPPORTED",
}
SIGNALS = {
    "registered_combined_behavioral": ("behavioral_combined", "calibrated_registered_probability"),
    "repeated_sampling_disagreement": ("repeated_disagreement", "raw_bounded_score"),
    "answer_self_consistency": ("sample_disagreement", "raw_bounded_score_alias_of_disagreement"),
    "verbalized_confidence": ("verbalized_confidence", "raw_bounded_risk_score"),
    "answer_stability": ("answer_stability_uncertainty", "raw_bounded_score_alias_of_disagreement"),
}
UNAVAILABLE_SIGNALS = {
    "semantic_response_consistency": "NOT_FEASIBLE_WITHOUT_NEW_UNREGISTERED_SEMANTIC_MODEL",
    "abstention_refusal": "NOT_IDENTIFIABLE_IN_SCHEMA_VALID_GROUP_ARTIFACTS",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_primary() -> None:
    registry_path = ROOT / "results/generated/final/rq5_scientific_claim_registry.json"
    manifest_path = ROOT / "results/generated/final/rq5_manifest.json"
    registry = read_json(registry_path)
    decision = registry["decisions"]["rq5_behavioral_selective_prediction"]
    actual = {
        "estimate": decision["estimate"],
        "ci_95": decision["ci_95"],
        "holm_adjusted_p": decision["holm_adjusted_p"],
        "decision": decision["decision"],
    }
    if actual != EXPECTED_PRIMARY:
        raise SystemExit("immutable RQ5 primary result drift")
    manifest = read_json(manifest_path)
    for binding in manifest["artifacts"]:
        path = ROOT / binding["path"]
        if not path.is_file() or sha256(path) != binding["sha256"]:
            raise SystemExit(f"RQ5 final manifest binding drift: {binding['path']}")


def tie_aware_curve(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(score, kind="stable")
    ys, scores = y[order].astype(float), score[order]
    expected = np.zeros(len(y), dtype=float)
    running = 0.0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and scores[end] == scores[start]:
            end += 1
        errors = float(ys[start:end].sum())
        width = end - start
        expected[start:end] = running + np.arange(1, width + 1) * errors / width
        running += errors
        start = end
    counts = np.arange(1, len(y) + 1, dtype=float)
    coverage = counts / len(y)
    risk = expected / counts
    return coverage, risk, float(risk.mean())


def risk_at(y: np.ndarray, score: np.ndarray, coverage: float) -> float:
    _, risk, _ = tie_aware_curve(y, score)
    index = min(len(risk) - 1, max(0, math.ceil(coverage * len(risk)) - 1))
    return float(risk[index])


def auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def auprc(y: np.ndarray, score: np.ndarray) -> float | None:
    positives = int(y.sum())
    if not positives:
        return None
    order = np.argsort(-score, kind="stable")
    ys, scores = y[order], score[order]
    tp = fp = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and scores[end] == scores[start]:
            end += 1
        tp += int(ys[start:end].sum())
        fp += int((end - start) - ys[start:end].sum())
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(area)


def calibration(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    p = np.clip(probability.astype(float), 1e-7, 1 - 1e-7)
    brier = float(np.mean((p - y) ** 2))
    nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    edges = np.linspace(0.0, 1.0, 16)
    ece = 0.0
    for index in range(15):
        left, right = edges[index], edges[index + 1]
        mask = (p >= left) & (p < right if index < 14 else p <= right)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return {"brier": brier, "nll": nll, "ece_15": ece}


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    _, _, aurc = tie_aware_curve(y, score)
    return {
        "aurc": aurc,
        "auroc": auroc(y, score),
        "auprc": auprc(y, score),
        **calibration(y, score),
        "risk_at_50_coverage": risk_at(y, score, 0.5),
        "risk_at_80_coverage": risk_at(y, score, 0.8),
    }


def stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = [rng.choice(np.flatnonzero(y == value), size=int(np.sum(y == value)), replace=True) for value in (0, 1)]
    return np.concatenate(parts)


def signal_score(row: dict, column: str) -> float:
    if column == "answer_stability_uncertainty":
        return float(1.0 - row["answer_stability"])
    return float(row[column])


def make_manifest(directory: Path, schema: str) -> None:
    artifacts = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name not in {"manifest.json", "manifest.json.sha256"}:
            artifacts.append({"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest = {
        "schema": schema,
        "status": "PASS",
        "classification": "SECONDARY_EXPLORATORY",
        "scientific_claim_authority": False,
        "primary_rq5_immutable": EXPECTED_PRIMARY,
        "artifacts": artifacts,
    }
    manifest_path = directory / "manifest.json"
    write_json(manifest_path, manifest)
    (directory / "manifest.json.sha256").write_text(sha256(manifest_path) + "\n", encoding="ascii", newline="\n")


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#404040",
            "text.color": "#171717",
            "axes.labelcolor": "#171717",
            "xtick.color": "#404040",
            "ytick.color": "#404040",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--output-base", default="results/generated/secondary")
    args = parser.parse_args()
    verify_primary()

    base = ROOT / args.output_base
    heterogeneity_dir = base / "rq5_heterogeneity"
    missingness_dir = base / "rq5_missingness"
    leaveout_dir = base / "rq5_leave_one_out"
    for directory in (heterogeneity_dir, missingness_dir, leaveout_dir):
        if directory.exists():
            raise SystemExit(f"refusing to overwrite secondary output: {directory}")
        directory.mkdir(parents=True)

    predictions = list(read_jsonl(ROOT / "results/rq5/analysis/heldout_predictions.jsonl"))
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in predictions:
        grouped[(row["model"], row["dataset"])].append(row)
    missing = read_json(ROOT / "results/rq5/analysis/missingness_sensitivity.json")
    missing_by_cell = {
        (row["model"], row["dataset"]): row
        for row in missing["per_cell_missingness"]
        if row["split"] == "test"
    }

    component_rows = []
    cell_effects = []
    rng = np.random.default_rng(20271101)
    bootstrap_by_cell: dict[tuple[str, str], np.ndarray] = {}
    for key in sorted(grouped):
        rows = grouped[key]
        y = np.asarray([row["error"] for row in rows], dtype=int)
        malformed = int(missing_by_cell[key]["malformed"])
        valid_responses = int(missing_by_cell[key]["schema_valid"])
        incomplete = int(sum(not row["complete_five"] for row in rows))
        common = {
            "model": key[0],
            "dataset": key[1],
            "correctness_prevalence": float(1.0 - y.mean()),
            "valid_group_count": len(rows),
            "valid_response_count": valid_responses,
            "malformed_response_count": malformed,
            "incomplete_group_count": incomplete,
        }
        for display, (column, interpretation) in SIGNALS.items():
            score = np.asarray([signal_score(row, column) for row in rows], dtype=float)
            component_rows.append(
                {
                    **common,
                    "signal": display,
                    "availability": "AVAILABLE",
                    "calibration_interpretation": interpretation,
                    **metrics(y, score),
                }
            )
        for display, reason in UNAVAILABLE_SIGNALS.items():
            component_rows.append(
                {
                    **common,
                    "signal": display,
                    "availability": reason,
                    "calibration_interpretation": "NOT_AVAILABLE",
                    **{name: None for name in ("aurc", "auroc", "auprc", "brier", "nll", "ece_15", "risk_at_50_coverage", "risk_at_80_coverage")},
                }
            )

        combined = np.asarray([row["behavioral_combined"] for row in rows], dtype=float)
        confidence = np.asarray([row["verbalized_confidence"] for row in rows], dtype=float)
        point = metrics(y, confidence)["aurc"] - metrics(y, combined)["aurc"]
        boot = []
        for _ in range(args.bootstrap_replicates):
            index = stratified_indices(y, rng)
            boot.append(metrics(y[index], confidence[index])["aurc"] - metrics(y[index], combined[index])["aurc"])
        bootstrap_by_cell[key] = np.asarray(boot)
        cell_effects.append(
            {
                "model": key[0],
                "dataset": key[1],
                "estimand": "AURC(verbalized_confidence)-AURC(registered_combined_behavioral)",
                "estimate": point,
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "bootstrap_replicates": args.bootstrap_replicates,
                "classification": "SECONDARY_EXPLORATORY",
            }
        )

    component_fields = [
        "model", "dataset", "signal", "availability", "calibration_interpretation", "aurc", "auroc", "auprc",
        "brier", "nll", "ece_15", "risk_at_50_coverage", "risk_at_80_coverage", "correctness_prevalence",
        "valid_group_count", "valid_response_count", "malformed_response_count", "incomplete_group_count",
    ]
    write_csv(heterogeneity_dir / "component_comparison.csv", component_rows, component_fields)
    write_json(heterogeneity_dir / "component_comparison.json", component_rows)
    available_display = [row for row in component_rows if row["availability"] == "AVAILABLE"]
    def rounded(value):
        return "" if value is None else f"{float(value):.4f}"
    write_csv(
        heterogeneity_dir / "component_ranking_metrics_display.csv",
        [
            {
                "model": row["model"], "dataset": row["dataset"], "signal": row["signal"],
                "aurc": rounded(row["aurc"]), "auroc": rounded(row["auroc"]), "auprc": rounded(row["auprc"]),
                "risk_at_50": rounded(row["risk_at_50_coverage"]), "risk_at_80": rounded(row["risk_at_80_coverage"]),
            }
            for row in available_display
        ],
        ["model", "dataset", "signal", "aurc", "auroc", "auprc", "risk_at_50", "risk_at_80"],
    )
    write_csv(
        heterogeneity_dir / "component_calibration_metrics_display.csv",
        [
            {
                "model": row["model"], "dataset": row["dataset"], "signal": row["signal"],
                "brier": rounded(row["brier"]), "nll": rounded(row["nll"]), "ece_15": rounded(row["ece_15"]),
                "correctness_prevalence": rounded(row["correctness_prevalence"]),
            }
            for row in available_display
        ],
        ["model", "dataset", "signal", "brier", "nll", "ece_15", "correctness_prevalence"],
    )
    accounting_rows = []
    for key in sorted(grouped):
        exemplar = next(row for row in component_rows if (row["model"], row["dataset"]) == key)
        accounting_rows.append(
            {
                "model": key[0], "dataset": key[1], "valid_groups": exemplar["valid_group_count"],
                "valid_responses": exemplar["valid_response_count"], "malformed_responses": exemplar["malformed_response_count"],
                "incomplete_groups": exemplar["incomplete_group_count"],
            }
        )
    write_csv(
        heterogeneity_dir / "component_sample_accounting_display.csv",
        accounting_rows,
        ["model", "dataset", "valid_groups", "valid_responses", "malformed_responses", "incomplete_groups"],
    )
    write_csv(
        heterogeneity_dir / "component_availability.csv",
        [{"signal": name, "status": reason} for name, reason in UNAVAILABLE_SIGNALS.items()],
        ["signal", "status"],
    )
    write_csv(
        heterogeneity_dir / "cell_effects.csv",
        cell_effects,
        ["model", "dataset", "estimand", "estimate", "ci_low", "ci_high", "bootstrap_replicates", "classification"],
    )
    write_csv(
        heterogeneity_dir / "cell_effects_display.csv",
        [
            {
                "model": row["model"], "dataset": row["dataset"], "delta_aurc": rounded(row["estimate"]),
                "bootstrap_95_ci": f"[{rounded(row['ci_low'])}, {rounded(row['ci_high'])}]",
            }
            for row in cell_effects
        ],
        ["model", "dataset", "delta_aurc", "bootstrap_95_ci"],
    )

    available_rows = [row for row in component_rows if row["availability"] == "AVAILABLE"]
    macro = {}
    for signal in SIGNALS:
        rows = [row for row in available_rows if row["signal"] == signal]
        macro[signal] = {
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("aurc", "auroc", "auprc", "brier", "nll", "ece_15", "risk_at_50_coverage", "risk_at_80_coverage")
        }
    write_json(
        heterogeneity_dir / "calibration_ranking_summary.json",
        {
            "schema": "trajuq_rq5_secondary_calibration_ranking_v1",
            "classification": "SECONDARY_EXPLORATORY",
            "scientific_claim_authority": False,
            "cell_weighting": "equal_model_dataset_cell_weight",
            "macro": macro,
            "alias_disclosure": {
                "answer_self_consistency": "same preserved sample_disagreement values as repeated_sampling_disagreement",
                "answer_stability": "one-minus-stability; algebraically identical to sample_disagreement in preserved artifacts",
            },
            "unavailable": UNAVAILABLE_SIGNALS,
        },
    )

    coverages = np.linspace(0.1, 1.0, 19)
    curve_rows = []
    curve_signals = {
        "registered combined behavioral": "behavioral_combined",
        "repeated-sampling disagreement": "repeated_disagreement",
        "verbalized confidence": "verbalized_confidence",
    }
    for label, column in curve_signals.items():
        for coverage in coverages:
            values = []
            for rows in grouped.values():
                y = np.asarray([row["error"] for row in rows], dtype=int)
                score = np.asarray([row[column] for row in rows], dtype=float)
                values.append(risk_at(y, score, float(coverage)))
            curve_rows.append({"signal": label, "coverage": float(coverage), "macro_risk": float(np.mean(values)), "n_cells": len(values)})
    write_csv(heterogeneity_dir / "coverage_risk_curves.csv", curve_rows, ["signal", "coverage", "macro_risk", "n_cells"])

    effects = np.asarray([row["estimate"] for row in cell_effects])
    write_json(
        heterogeneity_dir / "heterogeneity_summary.json",
        {
            "schema": "trajuq_rq5_secondary_heterogeneity_v1",
            "classification": "SECONDARY_EXPLORATORY",
            "scientific_claim_authority": False,
            "n_cells": len(cell_effects),
            "cells_favoring_combined": int(np.sum(effects > 0)),
            "cells_favoring_verbalized_confidence": int(np.sum(effects < 0)),
            "cells_tied": int(np.sum(effects == 0)),
            "effect_min": float(effects.min()),
            "effect_max": float(effects.max()),
            "effect_standard_deviation": float(effects.std(ddof=0)),
            "interpretation": "HETEROGENEITY_DESCRIPTIVE_NOT_A_CONFIRMATORY_GATE",
        },
    )

    complete_effects = []
    for key in sorted(grouped):
        rows = [row for row in grouped[key] if row["complete_five"]]
        if not rows:
            complete_effects.append(
                {
                    "model": key[0],
                    "dataset": key[1],
                    "n_groups": 0,
                    "aurc_improvement": None,
                    "status": "NOT_ESTIMABLE_NO_COMPLETE_FIVE_GROUPS",
                }
            )
            continue
        y = np.asarray([row["error"] for row in rows], dtype=int)
        combined = np.asarray([row["behavioral_combined"] for row in rows], dtype=float)
        confidence = np.asarray([row["verbalized_confidence"] for row in rows], dtype=float)
        complete_effects.append(
            {
                "model": key[0], "dataset": key[1], "n_groups": len(rows),
                "aurc_improvement": metrics(y, confidence)["aurc"] - metrics(y, combined)["aurc"],
                "status": "ESTIMATED",
            }
        )
    write_json(
        missingness_dir / "missingness_summary.json",
        {
            "schema": "trajuq_rq5_secondary_missingness_expanded_v1",
            "classification": "SECONDARY_EXPLORATORY",
            "scientific_claim_authority": False,
            "registered_valid_response_population": {
                "groups": len(predictions),
                "macro_aurc_improvement": float(np.mean(effects)),
            },
            "complete_five_response_groups": {
                "groups": int(sum(row["complete_five"] for row in predictions)),
                "macro_aurc_improvement": float(np.mean([row["aurc_improvement"] for row in complete_effects if row["aurc_improvement"] is not None])),
                "cells": complete_effects,
            },
            "conservative_malformed_response_sensitivity": {
                "status": "NOT_MATHEMATICALLY_IDENTIFIABLE",
                "reason": "malformed responses have neither a registered behavioral score nor a verbalized-confidence score; assigning either would introduce an unregistered imputation rule",
            },
            "per_cell_missingness": [missing_by_cell[key] for key in sorted(missing_by_cell)],
        },
    )

    leave_model = []
    models = sorted({key[0] for key in grouped})
    datasets = sorted({key[1] for key in grouped})
    for omitted in models:
        retained = [key for key in sorted(grouped) if key[0] != omitted]
        samples = np.mean(np.vstack([bootstrap_by_cell[key] for key in retained]), axis=0)
        leave_model.append(
            {"omitted_model": omitted, "retained_cells": len(retained), "estimate": float(np.mean([next(r["estimate"] for r in cell_effects if (r["model"], r["dataset"]) == key) for key in retained])), "ci_low": float(np.quantile(samples, 0.025)), "ci_high": float(np.quantile(samples, 0.975))}
        )
    leave_dataset = []
    for omitted in datasets:
        retained = [key for key in sorted(grouped) if key[1] != omitted]
        samples = np.mean(np.vstack([bootstrap_by_cell[key] for key in retained]), axis=0)
        leave_dataset.append(
            {"omitted_dataset": omitted, "retained_cells": len(retained), "estimate": float(np.mean([next(r["estimate"] for r in cell_effects if (r["model"], r["dataset"]) == key) for key in retained])), "ci_low": float(np.quantile(samples, 0.025)), "ci_high": float(np.quantile(samples, 0.975))}
        )
    write_json(
        leaveout_dir / "leave_one_out_summary.json",
        {
            "schema": "trajuq_rq5_secondary_leave_one_out_v1",
            "classification": "SECONDARY_EXPLORATORY",
            "scientific_claim_authority": False,
            "estimand": "equal-cell macro AURC(verbalized confidence)-AURC(combined behavioral)",
            "leave_one_model_out": leave_model,
            "leave_one_dataset_out": leave_dataset,
        },
    )

    configure_plotting()
    ordered = sorted(cell_effects, key=lambda row: (row["model"], row["dataset"]))
    short = {models[0]: "DeepSeek", models[1]: "Kimi", models[2]: "GLM"}
    labels = [f"{short[row['model']]} · {row['dataset'].replace('_', ' ')}" for row in ordered]
    estimates = np.asarray([row["estimate"] for row in ordered])
    lows = np.asarray([row["ci_low"] for row in ordered])
    highs = np.asarray([row["ci_high"] for row in ordered])
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(7.01, 6.10))
    ax.errorbar(estimates, positions, xerr=np.vstack([estimates - lows, highs - estimates]), fmt="o", color="#1F6F5F", ecolor="#8D8D88", markersize=4.2, elinewidth=0.8, capsize=2)
    ax.axvline(0, color="#171717", linewidth=0.8)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("ΔAURC = confidence − combined · positive favors combined")
    ax.set_title("Combined behavioral uncertainty varies across black-box cells", loc="left", fontweight="semibold")
    ax.grid(axis="x", color="#E4E7E5", linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, heterogeneity_dir / "rq5_cell_forest")

    fig, ax = plt.subplots(figsize=(7.01, 3.95))
    styles = {
        "registered combined behavioral": ("#1F6F5F", "-", 1.7),
        "verbalized confidence": ("#171717", "-", 1.2),
        "repeated-sampling disagreement": ("#8D8D88", "--", 1.2),
    }
    for label in ("registered combined behavioral", "verbalized confidence", "repeated-sampling disagreement"):
        rows = [row for row in curve_rows if row["signal"] == label]
        color, linestyle, width = styles[label]
        ax.plot([row["coverage"] for row in rows], [row["macro_risk"] for row in rows], label=label, color=color, linestyle=linestyle, linewidth=width)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk · lower is better")
    ax.set_title("No black-box signal dominates across the coverage range", loc="left", fontweight="semibold")
    ax.grid(axis="y", color="#E4E7E5", linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save_figure(fig, heterogeneity_dir / "rq5_coverage_risk")

    write_json(
        heterogeneity_dir / "visual_spec.json",
        {
            "claim": "RQ5 cell effects are heterogeneous and the confirmatory macro effect remains null",
            "visual_forms": ["dot-and-interval forest plot", "ordered coverage-risk line plot"],
            "metric": "AURC difference and selective risk",
            "direction": "positive delta favors combined; lower selective risk is better",
            "uncertainty": f"within-cell class-stratified percentile bootstrap, {args.bootstrap_replicates} replicates",
            "target_width_mm": 178,
            "output_formats": ["pdf", "svg", "png"],
            "classification": "SECONDARY_EXPLORATORY",
        },
    )

    make_manifest(heterogeneity_dir, "trajuq_rq5_secondary_heterogeneity_manifest_v1")
    make_manifest(missingness_dir, "trajuq_rq5_secondary_missingness_manifest_v1")
    make_manifest(leaveout_dir, "trajuq_rq5_secondary_leave_one_out_manifest_v1")
    print(json.dumps({"status": "PASS", "cells": len(grouped), "primary_unchanged": EXPECTED_PRIMARY, "outputs": [str(heterogeneity_dir), str(missingness_dir), str(leaveout_dir)]}, sort_keys=True))


if __name__ == "__main__":
    main()
