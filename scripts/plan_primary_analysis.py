#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plan all frozen primary analyses after the 18 scored model-dataset "
            "cells exist. This command does not inspect result values."
        )
    )
    parser.add_argument(
        "--primary-plan",
        default="results/primary_matrix_plan.json",
    )
    parser.add_argument(
        "--primary-config",
        default="configs/experiments/primary.yaml",
    )
    parser.add_argument(
        "--output-root",
        default="results/primary_analysis",
    )
    parser.add_argument(
        "--output",
        default="results/primary_analysis_plan.json",
    )
    args = parser.parse_args()

    plan_path = Path(args.primary_plan)
    if not plan_path.exists():
        raise SystemExit(f"Missing primary plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "PASS":
        raise SystemExit("Primary matrix plan is not PASS.")

    cfg = yaml.safe_load(Path(args.primary_config).read_text(encoding="utf-8"))
    expected_cells = int(plan.get("expected_cells", 0))
    cells = plan.get("cells", [])
    if expected_cells <= 0 or len(cells) != expected_cells:
        raise SystemExit(
            f"Primary plan cell count mismatch: {len(cells)} != {expected_cells}"
        )

    n_boot = int(cfg["statistics"]["bootstrap"]["n_boot"])
    seed = int(cfg["experiment"]["seed"])
    root = Path(args.output_root)

    analyses: list[dict] = []
    aggregation_args: list[str] = [
        "python",
        "scripts/aggregate_cells.py",
        "--n-boot",
        str(n_boot),
        "--seed",
        str(seed),
        "--primary-config",
        args.primary_config,
        "--output",
        str(root / "macro_primary.json"),
    ]

    seen_pairs: set[tuple[str, str]] = set()
    for cell in cells:
        adapter = str(cell["adapter"])
        dataset = str(cell["dataset"])
        pair = (adapter, dataset)
        if pair in seen_pairs:
            raise SystemExit(f"Duplicate primary cell: {adapter}:{dataset}")
        seen_pairs.add(pair)

        cal = str(cell["calibration"]["scored_output"])
        test = str(cell["test"]["scored_output"])
        cell_root = root / adapter / dataset
        metrics = cell_root / "metrics.json"
        predictions = cell_root / "predictions.jsonl"
        nested = cell_root / "nested_bootstrap.json"
        ablations = cell_root / "feature_ablation.json"
        raw_diag = cell_root / "raw_risk_diagnostic.json"

        evaluate = [
            "python",
            "scripts/evaluate_run.py",
            "--calibration",
            cal,
            "--test",
            test,
            "--seed",
            str(seed),
            "--output",
            str(metrics),
            "--predictions-output",
            str(predictions),
        ]
        compare_nested = [
            "python",
            "scripts/compare_calibrated.py",
            "--calibration",
            cal,
            "--test",
            test,
            "--n-boot",
            str(n_boot),
            "--seed",
            str(seed),
            "--output",
            str(nested),
        ]
        ablation = [
            "python",
            "scripts/evaluate_feature_ablation.py",
            "--calibration",
            cal,
            "--test",
            test,
            "--n-boot",
            str(n_boot),
            "--seed",
            str(seed),
            "--output",
            str(ablations),
        ]
        diagnostic = [
            "python",
            "scripts/compare_methods.py",
            "--input",
            test,
            "--n-boot",
            str(n_boot),
            "--seed",
            str(seed),
            "--output",
            str(raw_diag),
        ]

        label = f"{adapter}_{dataset}"
        aggregation_args.extend(["--cell", f"{label}={predictions}"])
        analyses.append(
            {
                "cell": f"{adapter}:{dataset}",
                "adapter": adapter,
                "dataset": dataset,
                "evaluate_argv": evaluate,
                "nested_comparison_argv": compare_nested,
                "feature_ablation_argv": ablation,
                "raw_risk_diagnostic_argv": diagnostic,
                "outputs": {
                    "metrics": str(metrics),
                    "predictions": str(predictions),
                    "nested_bootstrap": str(nested),
                    "feature_ablation": str(ablations),
                    "raw_risk_diagnostic": str(raw_diag),
                },
            }
        )

    macro_path = root / "macro_primary.json"
    gate_path = root / "rq1_claim_gate.json"
    gate_argv = [
        "python",
        "scripts/evaluate_rq1_claim_gate.py",
        "--macro",
        str(macro_path),
        "--primary-config",
        args.primary_config,
        "--output",
        str(gate_path),
    ]

    payload = {
        "protocol": "trajuq_primary_analysis_plan_v1",
        "status": "PASS",
        "primary_plan": str(plan_path),
        "primary_config": args.primary_config,
        "seed": seed,
        "n_boot": n_boot,
        "cell_count": len(analyses),
        "cell_analyses": analyses,
        "macro_aggregation_argv": aggregation_args,
        "rq1_claim_gate_argv": gate_argv,
        "final_outputs": {
            "macro_primary": str(macro_path),
            "rq1_claim_gate": str(gate_path),
        },
        "execution_order": (
            "Run all cell analyses first, then macro aggregation, then the RQ1 claim gate. "
            "Do not omit a weak cell or rerun with altered analysis settings."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(out),
                "cell_count": len(analyses),
                "macro_output": str(macro_path),
                "claim_gate_output": str(gate_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
