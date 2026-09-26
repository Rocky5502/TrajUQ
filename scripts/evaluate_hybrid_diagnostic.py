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


HYBRID_FEATURES = [
    "exact_text_disagreement",
    "parsed_answer_available",
    "parsed_answer_disagreement",
    "token_metrics_available",
    "normalized_token_edit_distance",
    "token_common_prefix_fraction",
    "normalized_length_difference",
]


def load_target(path: str) -> tuple[list[str], dict[str, dict], np.ndarray]:
    rows = list(read_jsonl(path))
    if not rows:
        raise SystemExit(f"Empty scored target: {path}")
    index = {str(row["example_id"]): row for row in rows}
    if len(index) != len(rows):
        raise SystemExit(f"Duplicate target IDs: {path}")
    if any(row.get("correct") is None for row in rows):
        raise SystemExit(f"Target must be scored: {path}")
    ids = [str(row["example_id"]) for row in rows]
    y = np.asarray(
        [0 if bool(index[item]["correct"]) else 1 for item in ids],
        dtype=int,
    )
    return ids, index, y


def load_diagnostic(path: str) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    if not rows:
        raise SystemExit(f"Empty hybrid diagnostic: {path}")
    index = {str(row["example_id"]): row for row in rows}
    if len(index) != len(rows):
        raise SystemExit(f"Duplicate diagnostic IDs: {path}")
    for item_id, row in index.items():
        meta = row.get("metadata", {})
        if meta.get("contains_correctness_label") is not False:
            raise SystemExit(
                f"{item_id}: hybrid diagnostic is not explicitly outcome-isolated"
            )
        if any(key in row for key in ("correct", "reference", "y_error")):
            raise SystemExit(f"{item_id}: outcome-like top-level field in diagnostic")
    return index


def matrix(
    ids: list[str],
    target: dict[str, dict],
    diagnostic: dict[str, dict],
    require_same_checkpoint: bool,
) -> np.ndarray:
    if set(ids) != set(diagnostic):
        raise SystemExit("Target/diagnostic ID sets differ")

    rows: list[list[float]] = []
    for item_id in ids:
        source = target[item_id]
        diag = diagnostic[item_id]
        if str(diag.get("dataset")) != str(source.get("dataset")):
            raise SystemExit(f"{item_id}: dataset mismatch")

        target_prompt = source.get("prompt_sha256")
        diag_prompt = diag.get("metadata", {}).get("prompt_sha256")
        if target_prompt and diag_prompt and target_prompt != diag_prompt:
            raise SystemExit(f"{item_id}: prompt hash mismatch")

        if require_same_checkpoint and (
            diag.get("metadata", {}).get("same_checkpoint_verified") is not True
        ):
            raise SystemExit(f"{item_id}: same-checkpoint diagnostic is not verified")

        features = diag.get("features", {})
        row = [float(features[name]) for name in HYBRID_FEATURES]
        if not np.all(np.isfinite(row)):
            raise SystemExit(f"{item_id}: non-finite hybrid diagnostic feature")
        rows.append(row)

    return np.asarray(rows, dtype=float)


def disagreement_groups(
    y_error: np.ndarray,
    x: np.ndarray,
) -> dict[str, float | int | None]:
    column = HYBRID_FEATURES.index("parsed_answer_disagreement")
    available_col = HYBRID_FEATURES.index("parsed_answer_available")
    available = x[:, available_col] > 0.5
    disagree = available & (x[:, column] > 0.5)
    agree = available & ~disagree

    def rate(mask: np.ndarray) -> float | None:
        return float(y_error[mask].mean()) if bool(mask.any()) else None

    agree_rate = rate(agree)
    disagree_rate = rate(disagree)
    delta = (
        None
        if agree_rate is None or disagree_rate is None
        else float(disagree_rate - agree_rate)
    )
    return {
        "parsed_answer_available_n": int(available.sum()),
        "agreement_n": int(agree.sum()),
        "disagreement_n": int(disagree.sum()),
        "agreement_error_rate": agree_rate,
        "disagreement_error_rate": disagree_rate,
        "disagreement_minus_agreement_error_rate": delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Held-out RQ4 diagnostic for same-checkpoint AR/diffusion disagreement. "
            "This is diagnostic only and is not part of the primary RQ1 endpoint."
        )
    )
    parser.add_argument("--cal-target", required=True)
    parser.add_argument("--cal-diagnostic", required=True)
    parser.add_argument("--test-target", required=True)
    parser.add_argument("--test-diagnostic", required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--require-same-checkpoint", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output", default=None)
    args = parser.parse_args()

    cal_ids, cal_target, y_cal = load_target(args.cal_target)
    test_ids, test_target, y_test = load_target(args.test_target)
    if set(cal_ids).intersection(test_ids):
        raise SystemExit("Calibration/test ID overlap detected")

    cal_diag = load_diagnostic(args.cal_diagnostic)
    test_diag = load_diagnostic(args.test_diagnostic)
    x_cal = matrix(
        cal_ids,
        cal_target,
        cal_diag,
        require_same_checkpoint=args.require_same_checkpoint,
    )
    x_test = matrix(
        test_ids,
        test_target,
        test_diag,
        require_same_checkpoint=args.require_same_checkpoint,
    )

    calibrator = RiskCalibrator(
        method="logistic",
        random_state=args.seed,
    ).fit(x_cal, y_cal)
    p_test = calibrator.predict_error_probability(x_test)

    det = error_detection_metrics(y_test, p_test)
    cal = calibration_metrics(y_test, p_test)
    curve = selective_curve(y_test, p_test)

    payload = {
        "protocol": "trajuq_hybrid_rq4_diagnostic_v1",
        "role": "RQ4_DIAGNOSTIC_NOT_PRIMARY_ENDPOINT",
        "selective_risk_version": SELECTIVE_RISK_VERSION,
        "features": HYBRID_FEATURES,
        "require_same_checkpoint": bool(args.require_same_checkpoint),
        "n_calibration": len(cal_ids),
        "n_test": len(test_ids),
        **det,
        **cal,
        "ece_10": expected_calibration_error(y_test, p_test, n_bins=10),
        "ece_20": expected_calibration_error(y_test, p_test, n_bins=20),
        "aurc": curve.aurc,
        "risk_at_50_coverage": risk_at_coverage(y_test, p_test, 0.50),
        "risk_at_80_coverage": risk_at_coverage(y_test, p_test, 0.80),
        "association": disagreement_groups(y_test, x_test),
        "warning": (
            "AR/diffusion disagreement is a pre-outcome diagnostic. "
            "Neither decoding path is treated as ground truth."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.predictions_output:
        pred = Path(args.predictions_output)
        pred.parent.mkdir(parents=True, exist_ok=True)
        with pred.open("w", encoding="utf-8") as handle:
            for item_id, y, p, feature_row in zip(
                test_ids,
                y_test,
                p_test,
                x_test,
            ):
                handle.write(
                    json.dumps(
                        {
                            "example_id": item_id,
                            "y_error": int(y),
                            "p_error_hybrid_disagreement": float(p),
                            "features": {
                                name: float(value)
                                for name, value in zip(HYBRID_FEATURES, feature_row)
                            },
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
