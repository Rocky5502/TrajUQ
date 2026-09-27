#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajuq.calibration import RiskCalibrator
from trajuq.evaluation import calibration_metrics, error_detection_metrics, selective_curve
from trajuq.features import FROZEN_FEATURE_GROUPS
from trajuq.io import read_jsonl
from trajuq.provenance import analysis_provenance, compact_row_binding
from trajuq.statistics import paired_bootstrap_improvement


def load_rows(path: str) -> list[dict]:
    rows = list(read_jsonl(path))
    ids = [str(r["example_id"]) for r in rows]
    if not rows:
        raise SystemExit(f"No rows in {path}")
    if len(ids) != len(set(ids)):
        raise SystemExit(f"Duplicate ids in {path}")
    if any(r.get("correct") is None for r in rows):
        raise SystemExit(f"Unscored rows in {path}")
    return rows


def xy(rows: list[dict], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [
            [
                float(row["generation"]["features"][name])
                for name in feature_names
            ]
            for row in rows
        ],
        dtype=float,
    )
    y = np.asarray([0 if bool(row["correct"]) else 1 for row in rows], dtype=int)
    if not np.all(np.isfinite(x)):
        raise SystemExit(f"Non-finite values for features={feature_names}")
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen feature-family ablations for TrajUQ RQ1."
    )
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-output", default=None)
    args = parser.parse_args()

    cal_rows = load_rows(args.calibration)
    test_rows = load_rows(args.test)
    cal_ids = [str(r["example_id"]) for r in cal_rows]
    test_ids = [str(r["example_id"]) for r in test_rows]
    if set(cal_ids).intersection(test_ids):
        raise SystemExit("Calibration/test ID overlap detected")

    probabilities: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}

    for group_name, feature_names in FROZEN_FEATURE_GROUPS.items():
        x_cal, y_cal = xy(cal_rows, feature_names)
        x_test, y_test = xy(test_rows, feature_names)
        if not np.array_equal(y_cal, xy(cal_rows, FROZEN_FEATURE_GROUPS["decision_only"])[1]):
            raise SystemExit("Calibration labels changed across feature views")
        model = RiskCalibrator(
            method="logistic",
            random_state=args.seed,
        ).fit(x_cal, y_cal)
        p = model.predict_error_probability(x_test)
        probabilities[group_name] = p

        det = error_detection_metrics(y_test, p)
        cal = calibration_metrics(y_test, p)
        curve = selective_curve(y_test, p)
        results[group_name] = {
            "features": feature_names,
            **det,
            **cal,
            "aurc": curve.aurc,
        }

    y_test = np.asarray(
        [0 if bool(row["correct"]) else 1 for row in test_rows],
        dtype=int,
    )
    baseline = probabilities["decision_only"]
    improvements: dict[str, dict] = {}
    for group_name, p in probabilities.items():
        if group_name == "decision_only":
            continue
        improvements[group_name] = {
            metric: paired_bootstrap_improvement(
                y_test,
                p,
                baseline,
                metric=metric,
                n_boot=args.n_boot,
                seed=args.seed,
            ).to_dict()
            for metric in ("auroc", "auprc", "aurc")
        }

    provenance = analysis_provenance(
        {
            "calibration": args.calibration,
            "test": args.test,
        }
    )
    row_binding = compact_row_binding(provenance)

    payload = {
        "protocol": "frozen_feature_family_ablation_v1",
        "provenance": provenance,
        "baseline": "decision_only",
        "delta_direction": "positive_means_group_is_better_than_decision_only",
        "group_order": list(FROZEN_FEATURE_GROUPS),
        "n_calibration": len(cal_rows),
        "n_test": len(test_rows),
        "groups": results,
        "paired_improvements_vs_decision_only": improvements,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.predictions_output:
        pred_out = Path(args.predictions_output)
        pred_out.parent.mkdir(parents=True, exist_ok=True)
        with pred_out.open("w", encoding="utf-8") as handle:
            for idx, row in enumerate(test_rows):
                handle.write(
                    json.dumps(
                        {
                            "example_id": str(row["example_id"]),
                            "y_error": int(y_test[idx]),
                            "correct": bool(row["correct"]),
                            "p_error_by_group": {
                                name: float(probabilities[name][idx])
                                for name in FROZEN_FEATURE_GROUPS
                            },
                            "analysis_provenance": row_binding,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        print(f"[TrajUQ] ablation predictions -> {pred_out}")


if __name__ == "__main__":
    main()
