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
    risk_at_coverage,
    selective_curve,
)
from trajuq.io import read_jsonl
from trajuq.repeated_sampling import vote_uncertainty
from trajuq.scoring import canonical_prediction


REPEATED_FEATURES = [
    "sample_disagreement",
    "sample_vote_entropy",
    "sample_unique_fraction",
]


def load_target(path: str) -> tuple[list[str], dict[str, dict], np.ndarray]:
    rows = list(read_jsonl(path))
    ids = [str(r["example_id"]) for r in rows]
    if not rows:
        raise SystemExit(f"No rows in {path}")
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate target IDs in {path}")
    if any(r.get("correct") is None for r in rows):
        raise SystemExit(f"Target artifact must be scored: {path}")
    return ids, {str(r["example_id"]): r for r in rows}, np.asarray(
        [0 if bool(r["correct"]) else 1 for r in rows],
        dtype=int,
    )


def load_sample_run(path: str) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    index = {str(r["example_id"]): r for r in rows}
    if len(index) != len(rows):
        raise SystemExit(f"Duplicate sample IDs in {path}")
    return index


def feature_matrix(
    target_ids: list[str],
    target: dict[str, dict],
    sample_paths: list[str],
) -> np.ndarray:
    if len(sample_paths) < 2:
        raise SystemExit("Provide at least two repeated sample runs.")

    sample_runs = [load_sample_run(path) for path in sample_paths]
    target_set = set(target_ids)
    for path, run in zip(sample_paths, sample_runs):
        if set(run) != target_set:
            raise SystemExit(f"Sample run ID set differs from target: {path}")

    rows: list[list[float]] = []
    for item_id in target_ids:
        dataset = str(target[item_id]["dataset"])
        target_prompt_hash = target[item_id].get("prompt_sha256")
        predictions: list[str | None] = []
        for path, run in zip(sample_paths, sample_runs):
            row = run[item_id]
            if str(row.get("dataset")) != dataset:
                raise SystemExit(f"Dataset mismatch for {item_id} in {path}")
            if target_prompt_hash and row.get("prompt_sha256") != target_prompt_hash:
                raise SystemExit(f"Prompt hash mismatch for {item_id} in {path}")
            if row.get("reference") is not None:
                raise SystemExit(f"Reference leaked into repeated-sampling artifact: {path}")
            predictions.append(
                canonical_prediction(
                    dataset,
                    str(row["generation"]["text"]),
                )
            )

        features = vote_uncertainty(predictions)
        rows.append([float(features[name]) for name in REPEATED_FEATURES])

    x = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(x)):
        raise SystemExit("Non-finite repeated-sampling features")
    return x


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrated repeated-sampling uncertainty baseline. Sample artifacts "
            "must be generated without access to correctness/reference labels."
        )
    )
    parser.add_argument("--cal-target", required=True)
    parser.add_argument("--cal-sample", action="append", required=True)
    parser.add_argument("--test-target", required=True)
    parser.add_argument("--test-sample", action="append", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.cal_sample) != len(args.test_sample):
        raise SystemExit("Calibration and test must use the same repeated-sample count.")

    cal_ids, cal_target, y_cal = load_target(args.cal_target)
    test_ids, test_target, y_test = load_target(args.test_target)
    if set(cal_ids).intersection(test_ids):
        raise SystemExit("Calibration/test ID overlap detected")

    x_cal = feature_matrix(cal_ids, cal_target, args.cal_sample)
    x_test = feature_matrix(test_ids, test_target, args.test_sample)

    calibrator = RiskCalibrator(
        method="logistic",
        random_state=args.seed,
    ).fit(x_cal, y_cal)
    p_test = calibrator.predict_error_probability(x_test)

    det = error_detection_metrics(y_test, p_test)
    cal = calibration_metrics(y_test, p_test)
    curve = selective_curve(y_test, p_test)

    payload = {
        "baseline": "repeated_sampling_vote_uncertainty",
        "features": REPEATED_FEATURES,
        "sample_count": len(args.cal_sample),
        "n_calibration": len(cal_ids),
        "n_test": len(test_ids),
        **det,
        **cal,
        "aurc": curve.aurc,
        "risk_at_50_coverage": risk_at_coverage(y_test, p_test, 0.50),
        "risk_at_80_coverage": risk_at_coverage(y_test, p_test, 0.80),
        "note": (
            "This baseline uses multiple label-free generations and therefore "
            "has higher inference cost than single-trajectory TrajUQ."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
