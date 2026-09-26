#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import numpy as np

from trajuq.adaptive import (
    bootstrap_two_budget_policy,
    evaluate_two_budget_policy,
    select_threshold_under_budget,
)
from trajuq.calibration import RiskCalibrator, cross_fitted_error_probabilities
from trajuq.features import PRIMARY_TRAJECTORY_FEATURES
from trajuq.io import read_jsonl


def aligned(base_path: str, refined_path: str):
    base_rows = list(read_jsonl(base_path))
    refined_rows = list(read_jsonl(refined_path))
    base = {str(r["example_id"]): r for r in base_rows}
    refined = {str(r["example_id"]): r for r in refined_rows}

    if len(base) != len(base_rows):
        raise SystemExit(f"Duplicate example IDs in {base_path}")
    if len(refined) != len(refined_rows):
        raise SystemExit(f"Duplicate example IDs in {refined_path}")
    if set(base) != set(refined):
        missing_refined = sorted(set(base) - set(refined))
        missing_base = sorted(set(refined) - set(base))
        raise SystemExit(
            "Base/refined ID sets differ: "
            f"missing_refined={len(missing_refined)} missing_base={len(missing_base)}"
        )

    ids = sorted(base)
    if not ids:
        raise SystemExit("No examples in base/refined runs")
    if any(base[i].get("correct") is None or refined[i].get("correct") is None for i in ids):
        raise SystemExit("Adaptive evaluation requires scored artifacts")

    x = np.asarray(
        [
            [
                float(base[i]["generation"]["features"][name])
                for name in PRIMARY_TRAJECTORY_FEATURES
            ]
            for i in ids
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(x)):
        raise SystemExit("NaN/Inf in base-budget trajectory features")

    base_correct = np.asarray([bool(base[i]["correct"]) for i in ids], dtype=float)
    refined_correct = np.asarray([bool(refined[i]["correct"]) for i in ids], dtype=float)
    y_error = 1 - base_correct.astype(int)
    return ids, x, y_error, base_correct, refined_correct


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibration-selected two-budget adaptive-compute evaluation using "
            "trajectory+decision calibrated error probability with a selective second-pass rerun."
        )
    )
    parser.add_argument("--cal-base", required=True)
    parser.add_argument("--cal-refined", required=True)
    parser.add_argument("--test-base", required=True)
    parser.add_argument("--test-refined", required=True)
    parser.add_argument("--max-refine-fraction", type=float, default=0.50)
    parser.add_argument("--base-compute", type=float, required=True)
    parser.add_argument("--refined-compute", type=float, required=True)
    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--cost-mode",
        choices=["selective_second_pass_rerun"],
        default="selective_second_pass_rerun",
        help=(
            "Primary protocol: base generation is always paid; selected examples "
            "launch a separate full refined-budget generation."
        ),
    )
    args = parser.parse_args()

    cal_ids, x_cal, y_cal, cal_base, cal_refined = aligned(
        args.cal_base,
        args.cal_refined,
    )
    cal_risk_oof, folds_used = cross_fitted_error_probabilities(
        x_cal,
        y_cal,
        n_splits=args.crossfit_folds,
        random_state=args.seed,
    )
    selected = select_threshold_under_budget(
        cal_risk_oof,
        cal_base,
        cal_refined,
        max_refine_fraction=args.max_refine_fraction,
        base_compute=args.base_compute,
        refined_compute=args.refined_compute,
        cost_mode=args.cost_mode,
    )

    test_ids, x_test, y_test, test_base, test_refined = aligned(
        args.test_base,
        args.test_refined,
    )
    if set(cal_ids).intersection(test_ids):
        raise SystemExit("Calibration/test ID overlap detected")

    final_calibrator = RiskCalibrator(
        method="logistic",
        random_state=args.seed,
    ).fit(x_cal, y_cal)
    test_risk = final_calibrator.predict_error_probability(x_test)

    test = evaluate_two_budget_policy(
        test_risk,
        test_base,
        test_refined,
        threshold=selected.threshold,
        base_compute=args.base_compute,
        refined_compute=args.refined_compute,
        cost_mode=args.cost_mode,
    )

    bootstrap = bootstrap_two_budget_policy(
        test_risk,
        test_base,
        test_refined,
        threshold=selected.threshold,
        base_compute=args.base_compute,
        refined_compute=args.refined_compute,
        n_boot=args.n_boot,
        seed=args.seed,
        cost_mode=args.cost_mode,
    )

    payload = {
        "risk_source": "trajectory_plus_decision_logistic",
        "execution_semantics": args.cost_mode,
        "policy_threshold_source": "out_of_fold_calibration_predictions",
        "crossfit_folds_used": folds_used,
        "selected_on_calibration": selected.to_dict(),
        "test": test.to_dict(),
        "paired_bootstrap": bootstrap.to_dict(),
        "n_calibration": len(cal_ids),
        "n_test": len(test_ids),
        "base_accuracy": float(test_base.mean()),
        "refined_accuracy": float(test_refined.mean()),
        "base_error_rate": float(y_test.mean()),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        from pathlib import Path

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
