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
from trajuq.provenance import analysis_provenance
from trajuq.repeated_sampling import equivalence_cluster_labels, vote_uncertainty
from trajuq.scoring import canonical_prediction, math500_equivalent


REPEATED_FEATURES = [
    "sample_disagreement",
    "sample_vote_entropy",
    "sample_unique_fraction",
]

SIGNATURE_METADATA_KEYS = [
    "model_revision",
    "requested_device",
    "requested_dtype",
    "mask_id",
    "steps_requested",
    "max_new_tokens",
    "gen_length",
    "block_length",
    "temperature",
    "top_p",
    "cfg_scale",
    "remasking",
    "algorithm",
    "algorithm_temperature",
    "denoising_steps",
    "top_k",
    "remasking_strategy",
    "confidence_threshold",
    "trajectory_semantics",
    "upstream_repository",
    "upstream_commit",
    "partial_block_transfer_guard",
]


def uncertainty_features_for_outputs(
    dataset: str,
    outputs: list[str],
) -> dict[str, float]:
    name = dataset.lower()
    if name == "math500":
        labels = equivalence_cluster_labels(outputs, math500_equivalent)
        return vote_uncertainty(labels)

    predictions = [
        canonical_prediction(name, output)
        for output in outputs
    ]
    return vote_uncertainty(predictions)


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


def run_signature(rows: list[dict], path: str) -> dict:
    if not rows:
        raise SystemExit(f"Empty repeated-sampling run: {path}")

    first = rows[0]
    meta = first.get("generation", {}).get("metadata", {})
    signature = {
        "seed": int(first["seed"]),
        "model": str(first["model"]),
        "adapter": str(first["adapter"]),
        **{key: meta.get(key) for key in SIGNATURE_METADATA_KEYS},
    }

    for row in rows:
        if row.get("correct") is not None:
            raise SystemExit(f"Correctness leaked into repeated-sampling artifact: {path}")
        if row.get("reference") is not None:
            raise SystemExit(f"Reference leaked into repeated-sampling artifact: {path}")
        row_meta = row.get("generation", {}).get("metadata", {})
        row_signature = {
            "seed": int(row["seed"]),
            "model": str(row["model"]),
            "adapter": str(row["adapter"]),
            **{key: row_meta.get(key) for key in SIGNATURE_METADATA_KEYS},
        }
        if row_signature != signature:
            raise SystemExit(f"Generation signature changes within sample run: {path}")
    return signature


def load_sample_run(path: str) -> tuple[dict[str, dict], dict]:
    rows = list(read_jsonl(path))
    index = {str(r["example_id"]): r for r in rows}
    if len(index) != len(rows):
        raise SystemExit(f"Duplicate sample IDs in {path}")
    return index, run_signature(rows, path)


def feature_matrix(
    target_ids: list[str],
    target: dict[str, dict],
    sample_paths: list[str],
) -> tuple[np.ndarray, list[dict], dict[str, float | None]]:
    if len(sample_paths) < 2:
        raise SystemExit("Provide at least two repeated sample runs.")

    loaded = [load_sample_run(path) for path in sample_paths]
    sample_runs = [item[0] for item in loaded]
    signatures = [item[1] for item in loaded]
    seeds = [int(sig["seed"]) for sig in signatures]
    if len(seeds) != len(set(seeds)):
        raise SystemExit("Repeated-sampling runs must have unique seeds.")

    non_seed_signatures = [
        {key: value for key, value in sig.items() if key != "seed"}
        for sig in signatures
    ]
    if any(sig != non_seed_signatures[0] for sig in non_seed_signatures[1:]):
        raise SystemExit("Repeated-sampling generation settings differ across seeds.")

    target_set = set(target_ids)
    for path, run in zip(sample_paths, sample_runs):
        if set(run) != target_set:
            raise SystemExit(f"Sample run ID set differs from target: {path}")

    rows: list[list[float]] = []
    auxiliary_seconds_per_example: list[float] = []

    for item_id in target_ids:
        dataset = str(target[item_id]["dataset"])
        target_prompt_hash = target[item_id].get("prompt_sha256")
        target_model = str(target[item_id].get("model"))
        outputs: list[str] = []
        total_seconds = 0.0
        seconds_complete = True

        for path, run in zip(sample_paths, sample_runs):
            row = run[item_id]
            if str(row.get("dataset")) != dataset:
                raise SystemExit(f"Dataset mismatch for {item_id} in {path}")
            if str(row.get("model")) != target_model:
                raise SystemExit(f"Model mismatch between target and sample run: {item_id} in {path}")
            if target_prompt_hash and row.get("prompt_sha256") != target_prompt_hash:
                raise SystemExit(f"Prompt hash mismatch for {item_id} in {path}")

            outputs.append(str(row["generation"]["text"]))
            wall = row.get("generation", {}).get("metadata", {}).get("wall_time_seconds")
            try:
                wall_value = float(wall)
            except (TypeError, ValueError):
                seconds_complete = False
            else:
                if not np.isfinite(wall_value) or wall_value < 0:
                    seconds_complete = False
                else:
                    total_seconds += wall_value

        features = uncertainty_features_for_outputs(dataset, outputs)
        rows.append([float(features[name]) for name in REPEATED_FEATURES])
        if seconds_complete:
            auxiliary_seconds_per_example.append(total_seconds)

    x = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(x)):
        raise SystemExit("Non-finite repeated-sampling features")

    cost = {
        "mean_auxiliary_wall_time_seconds_per_example": (
            float(np.mean(auxiliary_seconds_per_example))
            if len(auxiliary_seconds_per_example) == len(target_ids)
            else None
        ),
        "p90_auxiliary_wall_time_seconds_per_example": (
            float(np.quantile(auxiliary_seconds_per_example, 0.90))
            if len(auxiliary_seconds_per_example) == len(target_ids)
            else None
        ),
    }
    return x, signatures, cost


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
    parser.add_argument("--expected-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output", default=None)
    args = parser.parse_args()

    if len(args.cal_sample) != len(args.test_sample):
        raise SystemExit("Calibration and test must use the same repeated-sample count.")
    if len(args.cal_sample) != args.expected_samples:
        raise SystemExit(
            f"Expected {args.expected_samples} repeated samples; got {len(args.cal_sample)}."
        )

    cal_ids, cal_target, y_cal = load_target(args.cal_target)
    test_ids, test_target, y_test = load_target(args.test_target)
    if set(cal_ids).intersection(test_ids):
        raise SystemExit("Calibration/test ID overlap detected")

    x_cal, cal_signatures, cal_cost = feature_matrix(
        cal_ids,
        cal_target,
        args.cal_sample,
    )
    x_test, test_signatures, test_cost = feature_matrix(
        test_ids,
        test_target,
        args.test_sample,
    )

    cal_by_seed = {int(sig["seed"]): sig for sig in cal_signatures}
    test_by_seed = {int(sig["seed"]): sig for sig in test_signatures}
    if len(cal_by_seed) != len(cal_signatures):
        raise SystemExit("Calibration repeated-sampling seeds are not unique.")
    if len(test_by_seed) != len(test_signatures):
        raise SystemExit("Test repeated-sampling seeds are not unique.")
    if set(cal_by_seed) != set(test_by_seed):
        raise SystemExit("Calibration/test repeated-sampling seed sets differ.")
    for seed in sorted(cal_by_seed):
        cal_sig = {k: v for k, v in cal_by_seed[seed].items() if k != "seed"}
        test_sig = {k: v for k, v in test_by_seed[seed].items() if k != "seed"}
        if cal_sig != test_sig:
            raise SystemExit(
                f"Calibration/test generation settings differ for repeated seed {seed}."
            )

    calibrator = RiskCalibrator(
        method="logistic",
        random_state=args.seed,
    ).fit(x_cal, y_cal)
    p_test = calibrator.predict_error_probability(x_test)

    det = error_detection_metrics(y_test, p_test)
    cal = calibration_metrics(y_test, p_test)
    curve = selective_curve(y_test, p_test)

    pred_out: Path | None = None
    if args.predictions_output:
        pred_out = Path(args.predictions_output)
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        with pred_out.open("w", encoding="utf-8") as f:
            for item_id, y, p in zip(test_ids, y_test, p_test):
                f.write(
                    json.dumps(
                        {
                            "example_id": item_id,
                            "y_error": int(y),
                            "correct": bool(1 - y),
                            "p_error_repeated_sampling": float(p),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    provenance_sources: dict[str, str] = {
        "cal_target": args.cal_target,
        "test_target": args.test_target,
    }
    provenance_sources.update(
        {
            f"cal_sample:{index}": path
            for index, path in enumerate(args.cal_sample)
        }
    )
    provenance_sources.update(
        {
            f"test_sample:{index}": path
            for index, path in enumerate(args.test_sample)
        }
    )
    if pred_out is not None:
        provenance_sources["predictions"] = str(pred_out)

    payload = {
        "protocol": "trajuq_repeated_sampling_baseline_v2",
        "selective_risk_version": SELECTIVE_RISK_VERSION,
        "baseline": "repeated_sampling_symbolic_or_canonical_vote_uncertainty",
        "math500_clustering": "pairwise_math_verify_connected_components_v1",
        "features": REPEATED_FEATURES,
        "sample_count": len(args.cal_sample),
        "sample_seeds": sorted(cal_by_seed),
        "generation_signature": {
            k: v
            for k, v in cal_signatures[0].items()
            if k != "seed"
        },
        "n_calibration": len(cal_ids),
        "n_test": len(test_ids),
        **det,
        **cal,
        "ece_10": expected_calibration_error(y_test, p_test, n_bins=10),
        "ece_20": expected_calibration_error(y_test, p_test, n_bins=20),
        "aurc": curve.aurc,
        "risk_at_50_coverage": risk_at_coverage(y_test, p_test, 0.50),
        "risk_at_80_coverage": risk_at_coverage(y_test, p_test, 0.80),
        "auxiliary_compute": {
            "calibration": cal_cost,
            "test": test_cost,
            "note": (
                "Auxiliary repeated-generation wall time is extra UQ cost; "
                "the target generation itself is not counted here."
            ),
        },
        "provenance": analysis_provenance(provenance_sources),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if pred_out is not None:
        print(f"[TrajUQ] repeated-sampling predictions -> {pred_out}")


if __name__ == "__main__":
    main()
