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
from trajuq.features import FINAL_BASELINE_FEATURES, PRIMARY_TRAJECTORY_FEATURES
from trajuq.io import read_jsonl
from trajuq.provenance import analysis_provenance, compact_row_binding


def load_xy(path: str, names: list[str]):
    rows = list(read_jsonl(path))
    ids = [str(r["example_id"]) for r in rows]
    if not rows:
        raise SystemExit(f"No records in {path}")
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate ids in {path}")
    if any(r.get("correct") is None for r in rows):
        raise SystemExit(f"Unscored records in {path}")

    x = np.asarray(
        [[float(r["generation"]["features"][name]) for name in names] for r in rows],
        dtype=float,
    )
    y = np.asarray([0 if bool(r["correct"]) else 1 for r in rows], dtype=int)
    if not np.all(np.isfinite(x)):
        raise SystemExit(f"NaN/Inf feature in {path}")
    return ids, x, y


def fit_predict(
    cal_path: str,
    test_path: str,
    features: list[str],
    random_state: int,
):
    cal_ids, x_cal, y_cal = load_xy(cal_path, features)
    test_ids, x_test, y_test = load_xy(test_path, features)
    overlap = set(cal_ids).intersection(test_ids)
    if overlap:
        raise SystemExit(f"Calibration/test leakage: {len(overlap)} overlapping ids")

    calibrator = RiskCalibrator(
        "logistic",
        random_state=random_state,
    ).fit(x_cal, y_cal)
    p_test = calibrator.predict_error_probability(x_test)
    return cal_ids, test_ids, y_cal, y_test, p_test


def summarize(
    y_test: np.ndarray,
    p_test: np.ndarray,
    features: list[str],
    n_calibration: int,
) -> dict:
    det = error_detection_metrics(y_test, p_test)
    cal = calibration_metrics(y_test, p_test)
    curve = selective_curve(y_test, p_test)
    return {
        "features": features,
        "n_calibration": int(n_calibration),
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
    parser.add_argument("--predictions-output", default=None)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    cal_ids_b, test_ids_b, y_cal_b, y_test_b, p_base = fit_predict(
        args.calibration,
        args.test,
        FINAL_BASELINE_FEATURES,
        random_state=args.seed,
    )
    cal_ids_t, test_ids_t, y_cal_t, y_test_t, p_traj = fit_predict(
        args.calibration,
        args.test,
        PRIMARY_TRAJECTORY_FEATURES,
        random_state=args.seed,
    )

    if cal_ids_b != cal_ids_t or test_ids_b != test_ids_t:
        raise SystemExit("Feature views produced inconsistent example ordering")
    if not np.array_equal(y_cal_b, y_cal_t) or not np.array_equal(y_test_b, y_test_t):
        raise SystemExit("Correctness labels differ between feature views")

    provenance = analysis_provenance(
        {
            "calibration": args.calibration,
            "test": args.test,
        }
    )
    row_binding = compact_row_binding(provenance)

    results = {
        "protocol": "trajuq_heldout_evaluation_v1",
        "provenance": provenance,
        "selective_risk_version": SELECTIVE_RISK_VERSION,
        "seed": args.seed,
        "decision_entropy_baseline": summarize(
            y_test_b,
            p_base,
            FINAL_BASELINE_FEATURES,
            len(y_cal_b),
        ),
        "trajectory_plus_decision_logistic": summarize(
            y_test_t,
            p_traj,
            PRIMARY_TRAJECTORY_FEATURES,
            len(y_cal_t),
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))

    if args.predictions_output:
        test_meta_rows = list(read_jsonl(args.test))
        if len(test_meta_rows) != len(test_ids_b):
            raise SystemExit("Test metadata length differs from prediction length")
        meta_by_id = {str(row["example_id"]): row for row in test_meta_rows}
        if set(meta_by_id) != set(test_ids_b):
            raise SystemExit("Test metadata IDs differ from prediction IDs")

        pred_path = Path(args.predictions_output)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as f:
            for item_id, y, pb, pt in zip(test_ids_b, y_test_b, p_base, p_traj):
                source = meta_by_id[item_id]
                revision = (
                    source.get("generation", {}).get("metadata", {}).get("model_revision")
                    or source.get("metadata", {}).get("model_revision")
                )
                row = {
                    "example_id": item_id,
                    "dataset": source.get("dataset"),
                    "adapter": source.get("adapter"),
                    "model": source.get("model"),
                    "model_revision": revision,
                    "y_error": int(y),
                    "correct": bool(1 - y),
                    "p_error_decision_entropy": float(pb),
                    "p_error_trajuq": float(pt),
                    "analysis_provenance": row_binding,
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"[TrajUQ] predictions -> {pred_path}")


if __name__ == "__main__":
    main()
