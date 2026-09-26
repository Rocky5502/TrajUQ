#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import numpy as np

from trajuq.adaptive import evaluate_two_budget_policy, select_threshold_under_budget
from trajuq.io import read_jsonl
from trajuq.risk import RAW_RISK_VERSION, raw_trajectory_risk


def aligned(base_path: str, refined_path: str):
    base = {str(r["example_id"]): r for r in read_jsonl(base_path)}
    refined = {str(r["example_id"]): r for r in read_jsonl(refined_path)}
    ids = sorted(set(base).intersection(refined))
    if not ids:
        raise SystemExit("No overlapping example IDs between base/refined runs")
    if any(base[i].get("correct") is None or refined[i].get("correct") is None for i in ids):
        raise SystemExit("Adaptive evaluation requires scored artifacts")

    risk = np.asarray(
        [raw_trajectory_risk(base[i]["generation"]["features"]) for i in ids],
        dtype=float,
    )
    base_correct = np.asarray([bool(base[i]["correct"]) for i in ids], dtype=float)
    refined_correct = np.asarray([bool(refined[i]["correct"]) for i in ids], dtype=float)
    return ids, risk, base_correct, refined_correct


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibration-selected two-budget adaptive-compute evaluation."
    )
    parser.add_argument("--cal-base", required=True)
    parser.add_argument("--cal-refined", required=True)
    parser.add_argument("--test-base", required=True)
    parser.add_argument("--test-refined", required=True)
    parser.add_argument("--max-refine-fraction", type=float, default=0.50)
    parser.add_argument("--base-compute", type=float, required=True)
    parser.add_argument("--refined-compute", type=float, required=True)
    args = parser.parse_args()

    _, cal_risk, cal_base, cal_refined = aligned(args.cal_base, args.cal_refined)
    selected = select_threshold_under_budget(
        cal_risk,
        cal_base,
        cal_refined,
        max_refine_fraction=args.max_refine_fraction,
        base_compute=args.base_compute,
        refined_compute=args.refined_compute,
    )

    ids, test_risk, test_base, test_refined = aligned(args.test_base, args.test_refined)
    test = evaluate_two_budget_policy(
        test_risk,
        test_base,
        test_refined,
        threshold=selected.threshold,
        base_compute=args.base_compute,
        refined_compute=args.refined_compute,
    )

    payload = {
        "risk_version": RAW_RISK_VERSION,
        "selected_on_calibration": selected.to_dict(),
        "test": test.to_dict(),
        "n_test": len(ids),
        "base_accuracy": float(test_base.mean()),
        "refined_accuracy": float(test_refined.mean()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
