#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajuq.calibration import RiskCalibrator
from trajuq.evaluation import (
    calibration_metrics,
    error_detection_metrics,
    expected_calibration_error,
    risk_at_coverage,
    selective_curve,
)
from trajuq.io import read_jsonl


TRAJECTORY_FEATURES = [
    "entropy_final",
    "entropy_mean",
    "entropy_p90",
    "entropy_slope",
    "margin_mean",
    "margin_min",
    "js_mean",
    "js_p90",
    "js_slope",
    "late_js_mean",
    "revision_rate",
    "late_revision_rate",
    "mask_auc",
    "late_mask_fraction",
    "commitment_time_mean",
]

FINAL_BASELINE_FEATURES = ["entropy_final"]


def load_xy(path: str, names: list[str]):
    rows = list(read_jsonl(path))
    ids = [str(r["example_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate ids in {path}")
    if any(r.get("correct") is None for r in rows):
        raise SystemExit(f"Unscored records in {path}")
    x = np.asarray(
        [[float(r["generation"]["features"][name]) for name in names] for r in rows],
        dtype=float,
    )
    y = np.asarray([0 if r["correct"] else 1 for r in rows], dtype=int)
    if not np.all(np.isfinite(x)):
        raise SystemExit(f"NaN/Inf feature in {path}")
    return ids, x, y


def evaluate(cal_path: str, test_path: str, features: list[str]) -> dict:
    cal_ids, x_cal, y_cal = load_xy(cal_path, features)
    test_ids, x_test, y_test = load_xy(test_path, features)
    overlap = set(cal_ids).intersection(test_ids)
    if overlap:
        raise SystemExit(f"Calibration/test leakage: {len(overlap)} overlapping ids")

    calibrator = RiskCalibrator("logistic").fit(x_cal, y_cal)
    p_test = calibrator.predict_error_probability(x_test)
    det = error_detection_metrics(y_test, p_test)
    cal = calibration_metrics(y_test, p_test)
    curve = selective_curve(y_test, p_test)
    return {
        "features": features,
        "n_calibration": int(len(y_cal)),
        "n_test": int(len(y_test)),
        "error_rate_test": float(y_test.mean()),
        **det,
        **cal,
        "ece_10": expected_calibration_error(y_test, p_test, n_bins=10),
        "ece_20": expected_calibration_error(y_test, p_test, n_bins=20),
        "aurc": curve.aurc,
        "aurc_definition": "mean_selective_risk_over_k_over_n",
        "risk_at_50_coverage": risk_at_coverage(y_test, p_test, 0.50),
        "risk_at_80_coverage": risk_at_coverage(y_test, p_test, 0.80),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    results = {
        "final_entropy_baseline": evaluate(
            args.calibration,
            args.test,
            FINAL_BASELINE_FEATURES,
        ),
        "trajectory_plus_final_logistic": evaluate(
            args.calibration,
            args.test,
            TRAJECTORY_FEATURES,
        ),
    }
    Path(args.output).write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
