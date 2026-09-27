#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from trajuq.calibration import (
    RiskCalibrator,
    fit_isotonic_logistic_sensitivity,
)
from trajuq.evaluation import (
    SELECTIVE_RISK_VERSION,
    calibration_metrics,
    error_detection_metrics,
    expected_calibration_error,
    risk_at_coverage,
    selective_curve,
)
from trajuq.features import (
    FINAL_BASELINE_FEATURES,
    PRIMARY_TRAJECTORY_FEATURES,
)
from trajuq.io import read_jsonl, write_json_immutable
from trajuq.provenance import analysis_provenance


def load_xy(path: str, names: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows = list(read_jsonl(path))
    if not rows:
        raise SystemExit(f"No records in {path}")
    ids = [str(row.get("example_id", "")) for row in rows]
    if any(not item_id for item_id in ids):
        raise SystemExit(f"Missing example_id in {path}")
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate example_id in {path}")
    if any(row.get("correct") is None for row in rows):
        raise SystemExit(f"Unscored record in {path}")

    x = np.asarray(
        [
            [float(row["generation"]["features"][name]) for name in names]
            for row in rows
        ],
        dtype=float,
    )
    y = np.asarray(
        [0 if bool(row["correct"]) else 1 for row in rows],
        dtype=int,
    )
    if not np.all(np.isfinite(x)):
        raise SystemExit(f"NaN/Inf feature in {path}")
    return ids, x, y


def summarize(
    y_error: np.ndarray,
    p_error: np.ndarray,
) -> dict[str, float | str]:
    p = np.asarray(p_error, dtype=float)
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("calibrated probability outside [0,1]")
    det = error_detection_metrics(y_error, p)
    cal = calibration_metrics(y_error, p)
    curve = selective_curve(y_error, p)
    return {
        **det,
        **cal,
        "ece_10": expected_calibration_error(y_error, p, n_bins=10),
        "ece_20": expected_calibration_error(y_error, p, n_bins=20),
        "aurc": curve.aurc,
        "aurc_definition": "mean_selective_risk_over_k_over_n",
        "risk_at_50_coverage": risk_at_coverage(y_error, p, 0.50),
        "risk_at_80_coverage": risk_at_coverage(y_error, p, 0.80),
    }


def lower_is_better_delta(
    isotonic: dict[str, float | str],
    logistic: dict[str, float | str],
    name: str,
) -> float:
    return float(logistic[name]) - float(isotonic[name])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Supplementary calibration sensitivity using leak-safe cross-validated "
            "isotonic post-calibration of the same standardized logistic models."
        )
    )
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument(
        "--primary-config",
        default="configs/experiments/primary.yaml",
    )
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(
        Path(args.primary_config).read_text(encoding="utf-8")
    )
    calibration_cfg = cfg.get("calibration", {})
    if "isotonic" not in calibration_cfg.get("sensitivity", []):
        raise SystemExit("Frozen primary config does not register isotonic sensitivity.")
    iso_cfg = calibration_cfg.get("isotonic", {})
    if iso_cfg.get("base_estimator") != "standardized_logistic":
        raise SystemExit("Unexpected isotonic base estimator in frozen config.")
    if iso_cfg.get("ensemble") is not False:
        raise SystemExit("Frozen isotonic sensitivity requires ensemble=false.")

    cal_ids_b, x_cal_b, y_cal_b = load_xy(
        args.calibration,
        FINAL_BASELINE_FEATURES,
    )
    test_ids_b, x_test_b, y_test_b = load_xy(
        args.test,
        FINAL_BASELINE_FEATURES,
    )
    cal_ids_t, x_cal_t, y_cal_t = load_xy(
        args.calibration,
        PRIMARY_TRAJECTORY_FEATURES,
    )
    test_ids_t, x_test_t, y_test_t = load_xy(
        args.test,
        PRIMARY_TRAJECTORY_FEATURES,
    )
    if cal_ids_b != cal_ids_t or test_ids_b != test_ids_t:
        raise SystemExit("Feature views produced inconsistent example ordering.")
    if set(cal_ids_b).intersection(test_ids_b):
        raise SystemExit("Calibration/test example IDs overlap.")
    if not np.array_equal(y_cal_b, y_cal_t) or not np.array_equal(
        y_test_b,
        y_test_t,
    ):
        raise SystemExit("Feature views produced inconsistent correctness labels.")

    labels, counts = np.unique(y_cal_b, return_counts=True)
    class_counts = {
        str(int(label)): int(count)
        for label, count in zip(labels, counts)
    }
    min_class_count_observed = int(counts.min()) if len(counts) else 0
    min_samples = int(iso_cfg.get("min_calibration_samples", 1000))
    min_class_count = int(iso_cfg.get("min_class_count", 5))
    requested_folds = int(iso_cfg.get("cv_folds", 5))
    eligible = (
        len(labels) == 2
        and len(y_cal_b) >= min_samples
        and min_class_count_observed >= min_class_count
        and min_class_count_observed >= requested_folds
    )

    provenance = analysis_provenance(
        {
            "calibration": args.calibration,
            "test": args.test,
        }
    )
    base_payload = {
        "protocol": "trajuq_isotonic_calibration_sensitivity_v1",
        "role": "supplementary_calibration_sensitivity",
        "status": "PASS" if eligible else "SKIP_INSUFFICIENT_CALIBRATION",
        "provenance": provenance,
        "selective_risk_version": SELECTIVE_RISK_VERSION,
        "seed": int(args.seed),
        "n_calibration": int(len(y_cal_b)),
        "n_test": int(len(y_test_b)),
        "eligibility": {
            "eligible": bool(eligible),
            "min_calibration_samples_required": min_samples,
            "min_class_count_required": min_class_count,
            "cv_folds_requested": requested_folds,
            "observed_class_counts": class_counts,
            "observed_min_class_count": min_class_count_observed,
        },
        "method": {
            "base_estimator": "standardized_logistic",
            "isotonic_cv": "stratified_out_of_fold",
            "ensemble": False,
            "test_labels_used_for_fit": False,
        },
    }

    if eligible:
        logistic_base = RiskCalibrator(
            method="logistic",
            random_state=args.seed,
        ).fit(x_cal_b, y_cal_b)
        logistic_traj = RiskCalibrator(
            method="logistic",
            random_state=args.seed,
        ).fit(x_cal_t, y_cal_t)
        p_logistic_base = logistic_base.predict_error_probability(x_test_b)
        p_logistic_traj = logistic_traj.predict_error_probability(x_test_t)

        iso_base, folds_base = fit_isotonic_logistic_sensitivity(
            x_cal_b,
            y_cal_b,
            n_splits=requested_folds,
            random_state=args.seed,
        )
        iso_traj, folds_traj = fit_isotonic_logistic_sensitivity(
            x_cal_t,
            y_cal_t,
            n_splits=requested_folds,
            random_state=args.seed,
        )
        p_iso_base = np.asarray(
            iso_base.predict_proba(x_test_b)[:, 1],
            dtype=float,
        )
        p_iso_traj = np.asarray(
            iso_traj.predict_proba(x_test_t)[:, 1],
            dtype=float,
        )

        logistic_base_metrics = summarize(y_test_b, p_logistic_base)
        logistic_traj_metrics = summarize(y_test_b, p_logistic_traj)
        iso_base_metrics = summarize(y_test_b, p_iso_base)
        iso_traj_metrics = summarize(y_test_b, p_iso_traj)

        base_payload["effective_cv_folds"] = {
            "decision_entropy": int(folds_base),
            "trajectory_plus_decision": int(folds_traj),
        }
        base_payload["decision_entropy"] = {
            "primary_logistic": logistic_base_metrics,
            "isotonic_sensitivity": iso_base_metrics,
            "isotonic_minus_logistic_improvement": {
                name: lower_is_better_delta(
                    iso_base_metrics,
                    logistic_base_metrics,
                    name,
                )
                for name in (
                    "brier",
                    "nll",
                    "ece",
                    "ece_10",
                    "ece_20",
                    "aurc",
                    "risk_at_50_coverage",
                    "risk_at_80_coverage",
                )
            },
        }
        base_payload["trajectory_plus_decision"] = {
            "primary_logistic": logistic_traj_metrics,
            "isotonic_sensitivity": iso_traj_metrics,
            "isotonic_minus_logistic_improvement": {
                name: lower_is_better_delta(
                    iso_traj_metrics,
                    logistic_traj_metrics,
                    name,
                )
                for name in (
                    "brier",
                    "nll",
                    "ece",
                    "ece_10",
                    "ece_20",
                    "aurc",
                    "risk_at_50_coverage",
                    "risk_at_80_coverage",
                )
            },
        }

    out = Path(args.output)
    write_json_immutable(out, base_payload)
    print(json.dumps(base_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
