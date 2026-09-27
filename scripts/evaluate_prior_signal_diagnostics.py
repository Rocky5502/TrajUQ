#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajuq.calibration import RiskCalibrator
from trajuq.evaluation import (
    SELECTIVE_RISK_VERSION,
    calibration_metrics,
    error_detection_metrics,
    expected_calibration_error,
    risk_at_coverage,
    selective_curve,
)
from trajuq.io import read_jsonl


SIGNALS = {
    "trajectory_entropy": {
        "feature": "entropy_mean",
        "relation_to_prior_work": (
            "Prior-aligned average predictive entropy over the native "
            "denoising trajectory; TrajUQ uses normalized entropy only on "
            "active unresolved positions."
        ),
    },
    "top1_revision_rate": {
        "feature": "revision_rate",
        "relation_to_prior_work": (
            "Prior-aligned token-flip instability; TrajUQ normalizes top-1 "
            "prediction changes by eligible consecutive active-unresolved "
            "position observations."
        ),
    },
}


def load_rows(path: str) -> list[dict]:
    rows = list(read_jsonl(path))
    if not rows:
        raise SystemExit(f"No records in {path}")
    ids = [str(row.get("example_id", "")) for row in rows]
    if any(not item_id for item_id in ids):
        raise SystemExit(f"Missing example_id in {path}")
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate example_id in {path}")
    if any(row.get("correct") is None for row in rows):
        raise SystemExit(f"Unscored records in {path}")
    return rows


def labels(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [0 if bool(row["correct"]) else 1 for row in rows],
        dtype=int,
    )


def feature_values(rows: list[dict], feature: str) -> np.ndarray:
    values = np.asarray(
        [
            [float(row["generation"]["features"][feature])]
            for row in rows
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise SystemExit(f"NaN/Inf prior-aligned feature={feature}")
    return values


def summarize(
    y_test: np.ndarray,
    p_test: np.ndarray,
    *,
    feature: str,
    n_calibration: int,
) -> dict:
    det = error_detection_metrics(y_test, p_test)
    cal = calibration_metrics(y_test, p_test)
    curve = selective_curve(y_test, p_test)
    return {
        "feature": feature,
        "calibrator": "logistic",
        "n_calibration": int(n_calibration),
        "n_test": int(len(y_test)),
        "error_rate_test": float(y_test.mean()),
        **det,
        **cal,
        "ece_10": expected_calibration_error(
            y_test,
            p_test,
            n_bins=10,
        ),
        "ece_20": expected_calibration_error(
            y_test,
            p_test,
            n_bins=20,
        ),
        "aurc": curve.aurc,
        "aurc_definition": "mean_selective_risk_over_k_over_n",
        "risk_at_50_coverage": risk_at_coverage(
            y_test,
            p_test,
            0.50,
        ),
        "risk_at_80_coverage": risk_at_coverage(
            y_test,
            p_test,
            0.80,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate prior-aligned single-trajectory dLLM uncertainty "
            "signals from frozen TrajUQ telemetry. This is not a D-CoCoA "
            "reproduction."
        )
    )
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    cal_rows = load_rows(args.calibration)
    test_rows = load_rows(args.test)
    cal_ids = [str(row["example_id"]) for row in cal_rows]
    test_ids = [str(row["example_id"]) for row in test_rows]
    overlap = set(cal_ids).intersection(test_ids)
    if overlap:
        raise SystemExit(
            f"Calibration/test leakage: {len(overlap)} overlapping ids"
        )

    y_cal = labels(cal_rows)
    y_test = labels(test_rows)
    results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, spec in SIGNALS.items():
        feature = str(spec["feature"])
        x_cal = feature_values(cal_rows, feature)
        x_test = feature_values(test_rows, feature)
        calibrator = RiskCalibrator(
            "logistic",
            random_state=args.seed,
        ).fit(x_cal, y_cal)
        p_test = calibrator.predict_error_probability(x_test)
        predictions[name] = p_test
        results[name] = {
            "relation_to_prior_work": spec["relation_to_prior_work"],
            **summarize(
                y_test,
                p_test,
                feature=feature,
                n_calibration=len(cal_rows),
            ),
        }

    payload = {
        "protocol": "trajuq_prior_aligned_signal_diagnostic_v1",
        "role": "secondary_prior_aligned_signal_class_diagnostic",
        "not_a_reproduction_of": [
            "D-CoCoA-L",
            "D-CoCoA-G",
            "DiSE",
        ],
        "selective_risk_version": SELECTIVE_RISK_VERSION,
        "seed": args.seed,
        "n_calibration": len(cal_rows),
        "n_test": len(test_rows),
        "signal_order": list(SIGNALS),
        "signals": results,
        "scope_note": (
            "These diagnostics use signal classes already present in the "
            "native TrajUQ trace. They do not implement semantic trajectory "
            "dissimilarity, masked-likelihood Monte Carlo calls, or sequence "
            "regeneration required by distinct prior methods."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pred_path = Path(args.predictions_output)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(test_rows):
            record = {
                "example_id": str(row["example_id"]),
                "dataset": row.get("dataset"),
                "adapter": row.get("adapter"),
                "model": row.get("model"),
                "y_error": int(y_test[index]),
                "correct": bool(row["correct"]),
                "p_error_trajectory_entropy": float(
                    predictions["trajectory_entropy"][index]
                ),
                "p_error_top1_revision_rate": float(
                    predictions["top1_revision_rate"][index]
                ),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[TrajUQ] prior-aligned predictions -> {pred_path}")


if __name__ == "__main__":
    main()
