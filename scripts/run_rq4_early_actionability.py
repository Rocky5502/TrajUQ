#!/usr/bin/env python
"""Run frozen RQ4 early-actionability analysis from preserved token telemetry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


FRACTIONS = (0.25, 0.50, 0.75, 1.00)
FIELDS = ("selected_token_logprob", "entropy", "margin")
EXPECTED_PARENT_SHA = "814965b4af71b9e042098d161e5691b0722eccb07c831668932c0d87c00b7194"
EXPECTED_ADDENDUM_SHA = "b8ebc06ab5109524eda8da2ecb084a83f8d00ad757a84bc510baaa27b6c8b8d3"
EXPECTED_SCORED_SAMPLES_SHA = "1bb125b17b5fab5bd4ccc9c069044ab7f921f79180e28ea238e27146fd0d30b7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def linear_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    return float(np.dot(centered, values - values.mean()) / denominator) if denominator else 0.0


def trajectory_features(trace: list[dict], fraction: float) -> tuple[list[float], list[float]]:
    prefix_length = max(1, math.ceil(fraction * len(trace)))
    prefix = trace[:prefix_length]
    trajectory = []
    endpoint = []
    for field in FIELDS:
        values = np.asarray([float(step[field]) for step in prefix], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite token telemetry: {field}")
        trajectory.extend(
            [
                float(values[0]),
                float(values[-1]),
                float(values.mean()),
                float(values.min()),
                float(values.max()),
                linear_slope(values),
                float(values.std(ddof=0)),
            ]
        )
        endpoint.append(float(values[-1]))
    return trajectory, endpoint


def selective_curve(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(score, kind="stable")
    ys, scores = y[order].astype(float), score[order]
    cumulative = np.zeros(len(y), dtype=float)
    running = 0.0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and scores[end] == scores[start]:
            end += 1
        errors = float(ys[start:end].sum())
        width = end - start
        cumulative[start:end] = running + np.arange(1, width + 1) * errors / width
        running += errors
        start = end
    counts = np.arange(1, len(y) + 1, dtype=float)
    coverage = counts / len(y)
    risk = cumulative / counts
    return coverage, risk, float(risk.mean())


def risk_at(y: np.ndarray, score: np.ndarray, coverage: float) -> float:
    _, risk, _ = selective_curve(y, score)
    index = min(len(risk) - 1, max(0, math.ceil(coverage * len(risk)) - 1))
    return float(risk[index])


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    _, _, aurc = selective_curve(y, score)
    has_both_classes = len(np.unique(y)) == 2
    return {
        "aurc": aurc,
        "auroc": float(roc_auc_score(y, score)) if has_both_classes else None,
        "auprc": float(average_precision_score(y, score)) if has_both_classes else None,
        "risk_at_50_coverage": risk_at(y, score, 0.50),
        "risk_at_80_coverage": risk_at(y, score, 0.80),
        "class_dependent_metrics_estimable": has_both_classes,
    }


def fit_predict(x_cal: np.ndarray, y_cal: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, dict]:
    mean = x_cal.mean(axis=0)
    scale = x_cal.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    if len(np.unique(y_cal)) < 2:
        probability = float(y_cal.mean())
        return np.full(len(x_test), probability), {"model_type": "constant_single_class", "constant_probability": probability}
    model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000, random_state=2027)
    model.fit((x_cal - mean) / scale, y_cal)
    return model.predict_proba((x_test - mean) / scale)[:, 1], {
        "model_type": "standardized_l2_logistic_regression",
        "mean": mean.tolist(), "scale": scale.tolist(),
        "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
    }


def stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate(
        [rng.choice(np.flatnonzero(y == value), size=int(np.sum(y == value)), replace=True) for value in np.unique(y)]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--parent-contract", required=True)
    parser.add_argument("--addendum", required=True)
    parser.add_argument("--scored-samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    raw_root = Path(args.raw_root)
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    parent = Path(args.parent_contract)
    addendum = Path(args.addendum)
    if sha256(parent) != EXPECTED_PARENT_SHA:
        raise SystemExit("parent early-actionability contract drift")
    if sha256(addendum) != EXPECTED_ADDENDUM_SHA:
        raise SystemExit("implementation addendum drift")
    addendum_doc = json.loads(addendum.read_text(encoding="utf-8"))
    if addendum_doc["parent_contract_sha256"] != EXPECTED_PARENT_SHA:
        raise SystemExit("implementation addendum does not bind parent contract")

    scored_samples = Path(args.scored_samples)
    if sha256(scored_samples) != EXPECTED_SCORED_SAMPLES_SHA:
        raise SystemExit("scored-samples label source drift")
    scored_labels: dict[tuple[str, str, int, str], bool] = {}
    with scored_samples.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            key = (item["dataset"], item["split"], int(item["seed"]), item["example_id"])
            if key in scored_labels:
                raise SystemExit(f"duplicate scored-samples join key: {key}")
            if not isinstance(item.get("correct"), bool):
                raise SystemExit(f"non-boolean scored outcome: {key}")
            scored_labels[key] = item["correct"]
    if len(scored_labels) != 19420:
        raise SystemExit(f"expected 19420 scored labels, found {len(scored_labels)}")

    records: dict[tuple[str, str, float], list[dict]] = {}
    rows_seen = 0
    raw_files = sorted(raw_root.rglob("*.raw.jsonl"))
    if len(raw_files) != 60:
        raise SystemExit(f"expected 60 production shards, found {len(raw_files)}")
    for path in raw_files:
        split = path.parent.name
        if split not in {"calibration", "test"}:
            raise SystemExit(f"unexpected split path: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("adapter") != "llama_ar":
                    raise SystemExit("non-AR row in RQ4 production telemetry")
                if row.get("correct") is not None:
                    raise SystemExit("production raw correct field must remain null; use claim-bound scored labels")
                key = (row["dataset"], split, int(row["seed"]), row["example_id"])
                if key not in scored_labels:
                    raise SystemExit(f"production row has no scored label: {key}")
                error = int(not scored_labels.pop(key))
                trace = row.get("generation", {}).get("metadata", {}).get("token_trace")
                if not isinstance(trace, list) or not trace:
                    raise SystemExit("missing token trace")
                rows_seen += 1
                for fraction in FRACTIONS:
                    trajectory, endpoint = trajectory_features(trace, fraction)
                    records.setdefault((row["dataset"], split, fraction), []).append(
                        {
                            "error": error,
                            "trajectory": trajectory,
                            "endpoint": endpoint,
                        }
                    )
    if rows_seen != 19420:
        raise SystemExit(f"expected 19420 production rows, found {rows_seen}")
    if scored_labels:
        raise SystemExit(f"scored label source contains {len(scored_labels)} unmatched rows")

    datasets = sorted({key[0] for key in records})
    cell_metrics = []
    predictions: dict[tuple[str, float], dict] = {}
    calibrators = []
    for dataset in datasets:
        for fraction in FRACTIONS:
            cal = records[(dataset, "calibration", fraction)]
            test = records[(dataset, "test", fraction)]
            y_cal = np.asarray([row["error"] for row in cal], dtype=int)
            y_test = np.asarray([row["error"] for row in test], dtype=int)
            signals = {}
            for signal in ("trajectory", "endpoint"):
                x_cal = np.asarray([row[signal] for row in cal], dtype=float)
                x_test = np.asarray([row[signal] for row in test], dtype=float)
                score, model = fit_predict(x_cal, y_cal, x_test)
                signals[signal] = score
                calibrators.append({"dataset": dataset, "fraction": fraction, "signal": signal, "n_calibration": len(cal), **model})
                cell_metrics.append({"dataset": dataset, "fraction": fraction, "signal": signal, "n_test": len(test), **metrics(y_test, score)})
            predictions[(dataset, fraction)] = {"y": y_test, **signals}

    metric_names = ("aurc", "auroc", "auprc", "risk_at_50_coverage", "risk_at_80_coverage")
    macro = []
    for fraction in FRACTIONS:
        for signal in ("trajectory", "endpoint"):
            rows = [row for row in cell_metrics if row["fraction"] == fraction and row["signal"] == signal]
            summary = {"fraction": fraction, "signal": signal, "n_cells": len(rows)}
            for name in metric_names:
                values = [row[name] for row in rows if row[name] is not None]
                summary[name] = float(np.mean(values)) if values else None
                summary[f"n_estimable_{name}"] = len(values)
            macro.append(summary)

    rng = np.random.default_rng(2027)
    advantages = []
    for fraction in FRACTIONS:
        point_by_cell = []
        samples = []
        for dataset in datasets:
            pred = predictions[(dataset, fraction)]
            point_by_cell.append(metrics(pred["y"], pred["endpoint"])["aurc"] - metrics(pred["y"], pred["trajectory"])["aurc"])
        for _ in range(args.bootstrap_replicates):
            effects = []
            for dataset in datasets:
                pred = predictions[(dataset, fraction)]
                index = stratified_indices(pred["y"], rng)
                effects.append(metrics(pred["y"][index], pred["endpoint"][index])["aurc"] - metrics(pred["y"][index], pred["trajectory"][index])["aurc"])
            samples.append(float(np.mean(effects)))
        advantages.append(
            {
                "fraction": fraction,
                "estimand": "AURC(endpoint_prefix)-AURC(trajectory_prefix)",
                "estimate": float(np.mean(point_by_cell)),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_replicates": args.bootstrap_replicates,
            }
        )

    result = {
        "schema": "trajuq_rq4_early_actionability_secondary_v1",
        "status": "PASS",
        "classification": "SECONDARY_EXPLORATORY",
        "scientific_claim_authority": False,
        "production_rows": rows_seen,
        "production_shards": len(raw_files),
        "fractions": list(FRACTIONS),
        "datasets": datasets,
        "macro": macro,
        "trajectory_vs_endpoint": advantages,
        "single_class_policy": addendum_doc["evaluation"]["single_class_policy"],
        "interpretation_boundary": "fraction-wise descriptive analysis; no post-hoc earliest-useful threshold",
    }
    write_json_new(output / "early_actionability.json", result)
    write_json_new(output / "calibration_models.json", {"schema": "trajuq_rq4_early_actionability_calibrators_v1", "cells": calibrators})
    write_json_new(output / "cell_metrics.json", {"schema": "trajuq_rq4_early_actionability_cells_v1", "cells": cell_metrics})
    with (output / "macro_metrics.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "fraction", "signal", "n_cells",
                *[item for name in metric_names for item in (name, f"n_estimable_{name}")],
            ],
        )
        writer.writeheader()
        writer.writerows(macro)
    artifacts = []
    for path in sorted(output.iterdir()):
        if path.is_file():
            artifacts.append({"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size})
    manifest = {
        "schema": "trajuq_rq4_early_actionability_manifest_v1",
        "status": "PASS",
        "classification": "SECONDARY_EXPLORATORY",
        "scientific_claim_authority": False,
        "parent_contract_sha256": sha256(parent),
        "implementation_addendum_sha256": sha256(addendum),
        "scored_samples_sha256": sha256(scored_samples),
        "label_join": "exact_bijection_on_dataset_split_seed_example_id",
        "raw_root": str(raw_root),
        "artifacts": artifacts,
    }
    write_json_new(output / "manifest.json", manifest)
    (output / "manifest.json.sha256").write_text(sha256(output / "manifest.json") + "\n", encoding="ascii", newline="\n")
    print(json.dumps({"status": "PASS", "rows": rows_seen, "shards": len(raw_files), "output": str(output), "manifest_sha256": sha256(output / "manifest.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
