#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import numpy as np

from trajuq.calibration import RiskCalibrator
from trajuq.features import FINAL_BASELINE_FEATURES, PRIMARY_TRAJECTORY_FEATURES
from trajuq.io import read_jsonl
from trajuq.statistics import paired_bootstrap_improvement


def load(path: str, feature_names: list[str]):
    rows = list(read_jsonl(path))
    ids = [str(r["example_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate ids in {path}")
    if any(r.get("correct") is None for r in rows):
        raise SystemExit(f"Unscored rows in {path}")
    x = np.asarray(
        [[float(r["generation"]["features"][k]) for k in feature_names] for r in rows],
        dtype=float,
    )
    y = np.asarray([0 if bool(r["correct"]) else 1 for r in rows], dtype=int)
    if not np.all(np.isfinite(x)):
        raise SystemExit(f"NaN/Inf feature in {path}")
    return ids, x, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Primary RQ1 nested comparison: trajectory+decision logistic versus "
            "decision-entropy logistic on a frozen calibration/test pair."
        )
    )
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cal_ids_b, x_cal_b, y_cal_b = load(args.calibration, FINAL_BASELINE_FEATURES)
    test_ids_b, x_test_b, y_test_b = load(args.test, FINAL_BASELINE_FEATURES)
    cal_ids_t, x_cal_t, y_cal_t = load(args.calibration, PRIMARY_TRAJECTORY_FEATURES)
    test_ids_t, x_test_t, y_test_t = load(args.test, PRIMARY_TRAJECTORY_FEATURES)

    if cal_ids_b != cal_ids_t or test_ids_b != test_ids_t:
        raise SystemExit("Feature loaders produced inconsistent example ordering")
    if not np.array_equal(y_cal_b, y_cal_t) or not np.array_equal(y_test_b, y_test_t):
        raise SystemExit("Correctness labels differ between nested feature views")
    if set(cal_ids_b).intersection(test_ids_b):
        raise SystemExit("Calibration/test ID overlap detected")

    baseline = RiskCalibrator("logistic", random_state=args.seed).fit(x_cal_b, y_cal_b)
    candidate = RiskCalibrator("logistic", random_state=args.seed).fit(x_cal_t, y_cal_t)

    p_base = baseline.predict_error_probability(x_test_b)
    p_candidate = candidate.predict_error_probability(x_test_t)

    payload = {
        "protocol": "trajuq_nested_calibrated_comparison_v1",
        "comparison": "trajectory_plus_decision_logistic_vs_decision_entropy_logistic",
        "delta_direction": "positive_means_trajectory_augmented_is_better",
        "n_calibration": len(cal_ids_b),
        "n_test": len(test_ids_b),
        "metrics": {},
    }
    for metric in ("auroc", "auprc", "aurc"):
        payload["metrics"][metric] = paired_bootstrap_improvement(
            y_test_b,
            p_candidate,
            p_base,
            metric=metric,
            n_boot=args.n_boot,
            seed=args.seed,
        ).to_dict()

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        from pathlib import Path

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
