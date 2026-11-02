#!/usr/bin/env python
"""Fail-closed staged analysis of the frozen RQ5 black-box evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajuq.calibration import RiskCalibrator  # noqa: E402
from trajuq.evaluation import (  # noqa: E402
    calibration_metrics,
    error_detection_metrics,
    risk_at_coverage,
    selective_curve,
)
from trajuq.repeated_sampling import equivalence_cluster_labels, vote_uncertainty  # noqa: E402
from trajuq.scoring import canonical_prediction, math500_equivalent  # noqa: E402
from trajuq.statistics import stratified_bootstrap_indices  # noqa: E402


FEATURES = (
    "sample_disagreement",
    "sample_vote_entropy",
    "sample_unique_fraction",
    "verbalized_risk",
    "confidence_std",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def write_json(path: Path, value) -> None:
    write_new(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows) -> None:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    write_new(path, payload)


def require_plan(root: Path, plan_path: Path) -> dict:
    plan = read_json(plan_path)
    sidecar = Path(str(plan_path) + ".sha256").read_text(encoding="utf-8").strip()
    if sha256(plan_path) != sidecar:
        raise SystemExit("RQ5 plan hash drift")
    if plan.get("status") != "PASS" or plan.get("chronology") != "POST_DATA_PRE_OUTCOME_ANALYSIS":
        raise SystemExit("RQ5 plan is not eligible")
    for binding in plan["source_bindings"]:
        path = root / binding["path"]
        if sha256(path) != binding["sha256"]:
            raise SystemExit(f"RQ5 source binding drift: {binding['path']}")
    for binding in plan["input_bindings"]:
        path = root / binding["path"]
        if sha256(path) != binding["sha256"]:
            raise SystemExit(f"RQ5 input binding drift: {binding['path']}")
    return plan


def stage_validation(root: Path, plan: dict) -> None:
    registry_path = root / plan["inputs"]["response_registry"]
    ledger_path = root / plan["inputs"]["terminal_ledger"]
    categories = {}
    ledger_identity = {}
    for row in read_jsonl(ledger_path):
        request_uuid = row["request_uuid"]
        if request_uuid in categories:
            raise SystemExit("duplicate request UUID in reconciliation ledger")
        categories[request_uuid] = row["category"]
        ledger_identity[request_uuid] = (
            row["model"], row["dataset"], row["split"], str(row["example_id"]), int(row["repeat_index"])
        )
    if len(categories) != 58260 or Counter(categories.values()) != Counter(
        {"valid": 46681, "malformed": 11452, "provider_failure": 1, "ambiguous_locked": 126}
    ):
        raise SystemExit("terminal ledger accounting drift")

    valid_rows = []
    seen = set()
    registry_count = 0
    for row in read_jsonl(registry_path):
        registry_count += 1
        request_uuid = row["request_uuid"]
        if request_uuid in seen:
            raise SystemExit("duplicate UUID in response registry")
        seen.add(request_uuid)
        if request_uuid not in categories:
            raise SystemExit("registry UUID missing from terminal ledger")
        identity = (row["model"], row["dataset"], row["split"], str(row["example_id"]), int(row["repeat_index"]))
        if identity != ledger_identity[request_uuid]:
            raise SystemExit("registry/ledger identity mismatch")
        if categories[request_uuid] == "valid":
            if row.get("parsed_final_answer") in (None, "") or row.get("verbalized_confidence") is None:
                raise SystemExit("valid RQ5 row lacks a registered primary signal")
            valid_rows.append(row)
    if registry_count != 58133 or len(valid_rows) != 46681:
        raise SystemExit("validated response count drift")

    valid_rows.sort(key=lambda r: (r["model"], r["dataset"], r["split"], str(r["example_id"]), r["repeat_index"]))
    output = root / "results/rq5/analysis/valid_requests.jsonl"
    write_jsonl(output, valid_rows)
    write_json(
        root / "results/rq5/analysis/validation_summary.json",
        {
            "schema": "trajuq_rq5_validation_v1",
            "status": "PASS",
            "planned_requests": 58260,
            "saved_responses": registry_count,
            "schema_valid_requests": len(valid_rows),
            "malformed": 11452,
            "provider_failure": 1,
            "ambiguous_locked_no_replay": 126,
            "valid_requests_sha256": sha256(output),
            "claim_authorized": False,
        },
    )


def cluster_predictions(dataset: str, answers: list[str]) -> list[str]:
    if dataset == "math500":
        return equivalence_cluster_labels(answers, math500_equivalent)
    labels = []
    for answer in answers:
        label = canonical_prediction(dataset, f"FINAL: {answer}")
        if label is None:
            # A response may satisfy the frozen API schema while still failing
            # the narrower task parser (for example, a non-letter answer to a
            # multiple-choice item). Retain such valid responses as distinct,
            # deterministic normalized-text votes rather than dropping them.
            normalized = " ".join(str(answer).casefold().split())
            label = "<RAW>" + (normalized or "<EMPTY>")
        labels.append(label)
    return labels


def stage_features(root: Path, plan: dict) -> None:
    grouped = defaultdict(list)
    for row in read_jsonl(root / "results/rq5/analysis/valid_requests.jsonl"):
        key = (row["model"], row["dataset"], row["split"], str(row["example_id"]))
        grouped[key].append(row)
    output_rows = []
    excluded_lt2 = 0
    complete_groups = 0
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: int(row["repeat_index"]))
        if len(rows) < 2:
            excluded_lt2 += 1
            continue
        model, dataset, split, example_id = key
        labels = cluster_predictions(dataset, [str(row["parsed_final_answer"]) for row in rows])
        uncertainty = vote_uncertainty(labels)
        counts = Counter(labels)
        maximum = max(counts.values())
        modal = next(label for label in labels if counts[label] == maximum)
        representative = next(row for row, label in zip(rows, labels) if label == modal)
        confidences = np.asarray([float(row["verbalized_confidence"]) for row in rows]) / 100.0
        if len(rows) == 5:
            complete_groups += 1
        output_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "split": split,
                "example_id": example_id,
                "n_valid": len(rows),
                "n_registered": 5,
                "complete_five": len(rows) == 5,
                "error": int(not bool(representative["correct"])),
                **uncertainty,
                "verbalized_risk": float(1.0 - confidences.mean()),
                "confidence_std": float(confidences.std(ddof=0)),
                "answer_stability": float(1.0 - uncertainty["sample_disagreement"]),
            }
        )
    output = root / "results/rq5/analysis/group_features.jsonl"
    write_jsonl(output, output_rows)
    write_json(
        root / "results/rq5/analysis/feature_summary.json",
        {
            "schema": "trajuq_rq5_behavioral_features_v1",
            "status": "PASS",
            "features": list(FEATURES) + ["answer_stability"],
            "eligible_groups_n_valid_ge_2": len(output_rows),
            "complete_five_groups": complete_groups,
            "excluded_groups_n_valid_lt_2": excluded_lt2,
            "semantic_response_consistency": "NOT_FEASIBLE_WITHOUT_NEW_UNREGISTERED_SEMANTIC_MODEL",
            "group_features_sha256": sha256(output),
            "claim_authorized": False,
        },
    )


def feature_vector(row: dict) -> list[float]:
    return [float(row[name]) for name in FEATURES]


def stage_calibration(root: Path, plan: dict) -> None:
    grouped = defaultdict(list)
    for row in read_jsonl(root / "results/rq5/analysis/group_features.jsonl"):
        if row["split"] == "calibration":
            grouped[(row["model"], row["dataset"])].append(row)
    models = []
    for key in sorted(grouped):
        rows = grouped[key]
        x = np.asarray([feature_vector(row) for row in rows], dtype=float)
        y = np.asarray([row["error"] for row in rows], dtype=int)
        if len(np.unique(y)) < 2:
            models.append(
                {
                    "model": key[0],
                    "dataset": key[1],
                    "features": list(FEATURES),
                    "n_calibration": len(rows),
                    "model_type": "constant_empirical_error_single_class",
                    "constant_probability": float(y.mean()),
                    "calibration_class_counts": {
                        "correct": int(np.sum(y == 0)),
                        "error": int(np.sum(y == 1)),
                    },
                }
            )
        else:
            fitted = RiskCalibrator(method="logistic", random_state=2027).fit(x, y).model
            scale = fitted.named_steps["scale"]
            classifier = fitted.named_steps["clf"]
            models.append(
                {
                    "model": key[0],
                    "dataset": key[1],
                    "features": list(FEATURES),
                    "n_calibration": len(rows),
                    "model_type": "standardized_logistic_regression",
                    "mean": scale.mean_.tolist(),
                    "scale": scale.scale_.tolist(),
                    "coef": classifier.coef_[0].tolist(),
                    "intercept": float(classifier.intercept_[0]),
                }
            )
    if len(models) != 18:
        raise SystemExit("expected 18 RQ5 calibration cells")
    write_json(
        root / "results/rq5/analysis/calibration_models.json",
        {
            "schema": "trajuq_rq5_calibration_v1",
            "status": "PASS",
            "fit_partition": "calibration_only",
            "model_type": "standardized_logistic_regression_or_constant_for_single_class_cell",
            "random_state": 2027,
            "cells": models,
            "claim_authorized": False,
        },
    )


def predict_model(model: dict, row: dict) -> float:
    if model["model_type"] == "constant_empirical_error_single_class":
        return float(model["constant_probability"])
    x = np.asarray(feature_vector(row), dtype=float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coef = np.asarray(model["coef"], dtype=float)
    z = float(np.dot((x - mean) / scale, coef) + model["intercept"])
    return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z)))))


def metric_bundle(y: np.ndarray, risk: np.ndarray, probability: np.ndarray) -> dict:
    return {
        **error_detection_metrics(y, risk),
        "aurc": selective_curve(y, risk).aurc,
        "risk_at_50_coverage": risk_at_coverage(y, risk, 0.5),
        "risk_at_80_coverage": risk_at_coverage(y, risk, 0.8),
        **{("ece_15" if key == "ece" else key): value for key, value in calibration_metrics(y, probability, n_bins=15).items()},
    }


def stage_evaluation(root: Path, plan: dict) -> None:
    model_doc = read_json(root / "results/rq5/analysis/calibration_models.json")
    models = {(row["model"], row["dataset"]): row for row in model_doc["cells"]}
    predictions = []
    cells = defaultdict(list)
    for row in read_jsonl(root / "results/rq5/analysis/group_features.jsonl"):
        if row["split"] != "test":
            continue
        key = (row["model"], row["dataset"])
        combined = predict_model(models[key], row)
        prediction = {
            **row,
            "behavioral_combined": combined,
            "repeated_disagreement": float(row["sample_disagreement"]),
            "verbalized_confidence": float(row["verbalized_risk"]),
        }
        predictions.append(prediction)
        cells[key].append(prediction)
    metrics = []
    for key in sorted(cells):
        rows = cells[key]
        y = np.asarray([row["error"] for row in rows], dtype=int)
        if len(np.unique(y)) < 2:
            raise SystemExit(f"test cell lacks both classes: {key}")
        for signal in ("behavioral_combined", "repeated_disagreement", "verbalized_confidence"):
            risk = np.asarray([row[signal] for row in rows], dtype=float)
            metrics.append(
                {
                    "model": key[0],
                    "dataset": key[1],
                    "signal": signal,
                    "n": len(rows),
                    **metric_bundle(y, risk, np.clip(risk, 1e-7, 1 - 1e-7)),
                }
            )
    if len(metrics) != 54:
        raise SystemExit("expected 54 held-out metric rows")
    pred_path = root / "results/rq5/analysis/heldout_predictions.jsonl"
    write_jsonl(pred_path, predictions)
    write_json(
        root / "results/rq5/analysis/heldout_metrics.json",
        {
            "schema": "trajuq_rq5_heldout_v1",
            "status": "PASS",
            "partition": "test_only",
            "cells": metrics,
            "heldout_predictions_sha256": sha256(pred_path),
            "claim_authorized": False,
        },
    )


def stage_macro(root: Path, plan: dict) -> None:
    cells = read_json(root / "results/rq5/analysis/heldout_metrics.json")["cells"]
    grouped = defaultdict(list)
    metric_names = ("aurc", "auroc", "auprc", "brier", "nll", "ece_15", "risk_at_50_coverage", "risk_at_80_coverage")
    for row in cells:
        grouped[row["signal"]].append(row)
    macro = {
        signal: {name: float(np.mean([row[name] for row in rows])) for name in metric_names}
        for signal, rows in sorted(grouped.items())
    }
    write_json(
        root / "results/rq5/analysis/macro_aggregate.json",
        {
            "schema": "trajuq_rq5_macro_v1",
            "status": "PASS",
            "cell_weighting": "equal_model_dataset_cell_weight",
            "n_cells": 18,
            "macro": macro,
            "claim_authorized": False,
        },
    )


def aurc_improvement(rows: list[dict], indices: np.ndarray | None = None) -> float:
    y = np.asarray([row["error"] for row in rows], dtype=int)
    candidate = np.asarray([row["behavioral_combined"] for row in rows], dtype=float)
    baseline = np.asarray([row["verbalized_confidence"] for row in rows], dtype=float)
    if indices is not None:
        y, candidate, baseline = y[indices], candidate[indices], baseline[indices]
    return float(selective_curve(y, baseline).aurc - selective_curve(y, candidate).aurc)


def stage_bootstrap(root: Path, plan: dict) -> None:
    cells = defaultdict(list)
    for row in read_jsonl(root / "results/rq5/analysis/heldout_predictions.jsonl"):
        cells[(row["model"], row["dataset"])].append(row)
    point_by_cell = {key: aurc_improvement(rows) for key, rows in cells.items()}
    point = float(np.mean(list(point_by_cell.values())))
    model_point = {
        model: float(np.mean([value for (candidate, _), value in point_by_cell.items() if candidate == model]))
        for model in sorted({key[0] for key in cells})
    }
    heterogeneity_point = max(model_point.values()) - min(model_point.values())
    rng = np.random.default_rng(2027)
    h5a_samples = []
    h5b_samples = []
    for _ in range(10000):
        replicate = {}
        for key, rows in cells.items():
            y = np.asarray([row["error"] for row in rows], dtype=int)
            idx = stratified_bootstrap_indices(y, rng)
            replicate[key] = aurc_improvement(rows, idx)
        h5a_samples.append(float(np.mean(list(replicate.values()))))
        by_model = [
            float(np.mean([value for (candidate, _), value in replicate.items() if candidate == model]))
            for model in sorted(model_point)
        ]
        h5b_samples.append(max(by_model) - min(by_model))
    h5a_array = np.asarray(h5a_samples)
    h5b_array = np.asarray(h5b_samples)
    p_value = float(2 * min((np.sum(h5a_array <= 0) + 1) / 10001, (np.sum(h5a_array >= 0) + 1) / 10001))
    write_json(
        root / "results/rq5/analysis/paired_bootstrap.json",
        {
            "schema": "trajuq_rq5_bootstrap_v1",
            "status": "PASS",
            "bootstrap": "paired_class_stratified_within_cell_equal_cell_macro",
            "replicates": 10000,
            "seed": 2027,
            "hypotheses": {
                "rq5_behavioral_selective_prediction": {
                    "estimand": "AURC(verbalized_confidence)-AURC(behavioral_combined)",
                    "estimate": point,
                    "ci_low": float(np.quantile(h5a_array, 0.025)),
                    "ci_high": float(np.quantile(h5a_array, 0.975)),
                    "p_value_two_sided": p_value,
                    "n_cells": 18,
                },
                "rq5_cross_model_heterogeneity": {
                    "estimand": "range_of_model_macro_AURC_improvements",
                    "estimate": heterogeneity_point,
                    "ci_low": float(np.quantile(h5b_array, 0.025)),
                    "ci_high": float(np.quantile(h5b_array, 0.975)),
                    "p_value_two_sided": None,
                    "inference": "DESCRIPTIVE_BOOTSTRAP_ONLY_NO_VALID_ZERO_NULL_CALIBRATION",
                    "model_estimates": model_point,
                },
                "rq5_observability_gap": {
                    "estimate": None,
                    "p_value_two_sided": None,
                    "inference": "NOT_IDENTIFIABLE_MODELS_UNMATCHED_ACROSS_OBSERVABILITY_REGIMES",
                },
            },
            "claim_authorized": False,
        },
    )


def holm(raw: dict[str, float], alpha: float = 0.05) -> dict:
    ordered = sorted(raw, key=raw.get)
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * raw[name]))
        adjusted[name] = {"raw_p": raw[name], "adjusted_p": running, "reject": running <= alpha}
    return adjusted


def stage_multiplicity(root: Path, plan: dict) -> None:
    bootstrap = read_json(root / "results/rq5/analysis/paired_bootstrap.json")
    raw = {
        name: row["p_value_two_sided"]
        for name, row in bootstrap["hypotheses"].items()
        if row.get("p_value_two_sided") is not None
    }
    write_json(
        root / "results/rq5/analysis/multiplicity.json",
        {
            "schema": "trajuq_rq5_multiplicity_v1",
            "status": "PASS",
            "method": "holm_bonferroni_over_estimable_confirmatory_hypotheses",
            "alpha": 0.05,
            "family": list(plan["registered_science"]["confirmatory_family"]),
            "non_estimable_retained": [name for name in plan["registered_science"]["confirmatory_family"] if name not in raw],
            "decisions": holm(raw),
            "claim_authorized": False,
        },
    )


def stage_gate(root: Path, plan: dict) -> None:
    bootstrap = read_json(root / "results/rq5/analysis/paired_bootstrap.json")
    multiplicity = read_json(root / "results/rq5/analysis/multiplicity.json")
    h5a = bootstrap["hypotheses"]["rq5_behavioral_selective_prediction"]
    corrected = multiplicity["decisions"]["rq5_behavioral_selective_prediction"]
    supported = corrected["reject"] and h5a["ci_low"] > 0
    write_json(
        root / "results/rq5/analysis/hypothesis_gate.json",
        {
            "schema": "trajuq_rq5_hypothesis_gate_v1",
            "status": "PASS",
            "claim_authorized": True,
            "retain_positive_negative_null_heterogeneous": True,
            "decisions": {
                "rq5_behavioral_selective_prediction": {
                    "decision": "SUPPORTED" if supported else "NOT_SUPPORTED",
                    "estimate": h5a["estimate"],
                    "ci_95": [h5a["ci_low"], h5a["ci_high"]],
                    "holm_adjusted_p": corrected["adjusted_p"],
                },
                "rq5_cross_model_heterogeneity": {
                    "decision": "DESCRIPTIVE_ONLY",
                    **bootstrap["hypotheses"]["rq5_cross_model_heterogeneity"],
                },
                "rq5_observability_gap": {
                    "decision": "NOT_IDENTIFIABLE",
                    "reason": "models are unmatched across observability regimes",
                },
            },
        },
    )


def stage_sensitivity(root: Path, plan: dict) -> None:
    predictions = list(read_jsonl(root / "results/rq5/analysis/heldout_predictions.jsonl"))
    complete = [row for row in predictions if row["complete_five"]]
    by_cell = defaultdict(list)
    for row in complete:
        by_cell[(row["model"], row["dataset"])].append(row)
    complete_effects = {"|".join(key): aurc_improvement(rows) for key, rows in sorted(by_cell.items()) if len({r["error"] for r in rows}) == 2}
    all_cell_effects = {"|".join(key): aurc_improvement(rows) for key, rows in sorted(defaultdict(list, {
        key: [r for r in predictions if (r["model"], r["dataset"]) == key]
        for key in {(r["model"], r["dataset"]) for r in predictions}
    }).items())}
    models = sorted({row["model"] for row in predictions})
    datasets = sorted({row["dataset"] for row in predictions})
    leave_one_model_out = {
        model: float(np.mean([value for key, value in all_cell_effects.items() if not key.startswith(model + "|")]))
        for model in models
    }
    leave_one_dataset_out = {
        dataset: float(np.mean([value for key, value in all_cell_effects.items() if not key.endswith("|" + dataset)]))
        for dataset in datasets
    }
    reconciliation = read_json(root / plan["inputs"]["terminal_reconciliation"])
    write_json(
        root / "results/rq5/analysis/missingness_sensitivity.json",
        {
            "schema": "trajuq_rq5_secondary_missingness_v1",
            "status": "PASS",
            "classification": "SECONDARY_EXPLORATORY",
            "primary_valid_response_groups_n": len(predictions),
            "complete_five_groups_n": len(complete),
            "complete_five_cell_effects": complete_effects,
            "conservative_malformed_as_failure": "NOT_APPLIED_TO_DISAGREEMENT_WHERE_NO_RESPONSE_VALUE_EXISTS",
            "leave_one_model_out_macro_effect": leave_one_model_out,
            "leave_one_dataset_out_macro_effect": leave_one_dataset_out,
            "per_cell_missingness": reconciliation["cell_missingness"],
            "claim_authorized": False,
        },
    )


def stage_finalize(root: Path, plan: dict, plan_path: Path) -> None:
    artifacts = [
        "results/rq5/analysis/validation_summary.json",
        "results/rq5/analysis/feature_summary.json",
        "results/rq5/analysis/calibration_models.json",
        "results/rq5/analysis/heldout_metrics.json",
        "results/rq5/analysis/macro_aggregate.json",
        "results/rq5/analysis/paired_bootstrap.json",
        "results/rq5/analysis/multiplicity.json",
        "results/rq5/analysis/hypothesis_gate.json",
        "results/rq5/analysis/missingness_sensitivity.json",
    ]
    gate = read_json(root / "results/rq5/analysis/hypothesis_gate.json")
    claim_registry = {
        "schema": "trajuq_rq5_scientific_claim_registry_v1",
        "status": "PASS",
        "claim_authorized": True,
        "chronology": "POST_DATA_PRE_OUTCOME_ANALYSIS",
        "plan_sha256": sha256(plan_path),
        "terminal_ledger_sha256": plan["terminal_ledger_sha256"],
        "decisions": gate["decisions"],
        "bindings": [{"path": path, "sha256": sha256(root / path)} for path in artifacts],
        "authorized_interpretation": "Limited to gate decisions; unmatched cross-observability effects remain non-identifiable.",
    }
    claim_path = root / "results/generated/final/rq5_scientific_claim_registry.json"
    write_json(claim_path, claim_registry)
    manifest = {
        "schema": "trajuq_rq5_final_artifact_manifest_v1",
        "status": "FINAL",
        "claim_authorized": True,
        "plan_sha256": sha256(plan_path),
        "source_registry_sha256": plan["response_registry_sha256"],
        "terminal_ledger_sha256": plan["terminal_ledger_sha256"],
        "artifacts": claim_registry["bindings"] + [
            {"path": claim_path.relative_to(root).as_posix(), "sha256": sha256(claim_path)}
        ],
    }
    write_json(root / "results/generated/final/rq5_manifest.json", manifest)


STAGES = {
    "validation": stage_validation,
    "feature_extraction": stage_features,
    "calibration_fit": stage_calibration,
    "heldout_evaluation": stage_evaluation,
    "macro_aggregate": stage_macro,
    "paired_bootstrap": stage_bootstrap,
    "multiplicity_correction": stage_multiplicity,
    "hypothesis_gate": stage_gate,
    "missingness_sensitivity": stage_sensitivity,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=Path("results/rq5/rq5_downstream_plan.json"))
    parser.add_argument("--stage", choices=[*STAGES, "finalize"], required=True)
    args = parser.parse_args()
    root = ROOT
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    plan = require_plan(root, plan_path)
    if args.stage == "finalize":
        stage_finalize(root, plan, plan_path)
    else:
        STAGES[args.stage](root, plan)


if __name__ == "__main__":
    main()
