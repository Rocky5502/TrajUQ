#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def replace_arg(argv: list[str], flag: str, value: str) -> list[str]:
    out = list(argv)
    if flag not in out:
        out.extend([flag, value])
        return out
    idx = out.index(flag)
    if idx + 1 >= len(out):
        raise ValueError(f"Malformed argv: {flag} has no value")
    out[idx + 1] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plan low-budget base runs and adaptive evaluations from a frozen "
            "primary-matrix plan. Does not execute GPU work."
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
        "--base-run-root",
        default="runs/adaptive_base",
    )
    parser.add_argument(
        "--output",
        default="results/adaptive_matrix_plan.json",
    )
    args = parser.parse_args()

    primary_plan_path = Path(args.primary_plan)
    if not primary_plan_path.exists():
        raise SystemExit(f"Missing primary matrix plan: {primary_plan_path}")
    primary_plan = json.loads(primary_plan_path.read_text(encoding="utf-8"))
    if primary_plan.get("status") != "PASS":
        raise SystemExit("Primary matrix plan is not PASS.")

    cfg = yaml.safe_load(Path(args.primary_config).read_text(encoding="utf-8"))
    adaptive = cfg["adaptive_compute"]
    if adaptive.get("execution_semantics") != "selective_second_pass_rerun":
        raise SystemExit("Primary adaptive execution semantics changed unexpectedly.")
    primary_fraction = float(adaptive["primary_max_refine_fraction"])
    budgets = adaptive["budgets"]
    n_boot = int(adaptive["paired_bootstrap_n"])
    seed = int(cfg["experiment"]["seed"])
    folds = int(adaptive["crossfit_folds"])

    cells: list[dict] = []
    for cell in primary_plan.get("cells", []):
        adapter = str(cell["adapter"])
        dataset = str(cell["dataset"])
        if adapter not in budgets:
            raise SystemExit(f"No frozen adaptive budget for {adapter}")
        budget = budgets[adapter]

        base_partitions: dict[str, dict] = {}
        for partition in ("calibration", "test"):
            refined = cell[partition]
            generate = list(refined["generate_argv"])
            base_root = Path(args.base_run_root) / adapter / dataset
            raw = base_root / f"{partition}.raw.jsonl"
            scored = base_root / f"{partition}.scored.jsonl"

            generate = replace_arg(generate, "--output", str(raw))
            if adapter in {"llada", "dream"}:
                base_compute = float(budget["base_steps"])
                refined_compute = float(budget["refined_steps"])
                generate = replace_arg(
                    generate,
                    "--steps",
                    str(int(budget["base_steps"])),
                )
            elif adapter == "sdar":
                base_compute = float(budget["base_denoising_steps"])
                refined_compute = float(budget["refined_denoising_steps"])
                generate = replace_arg(
                    generate,
                    "--sdar-denoising-steps",
                    str(int(budget["base_denoising_steps"])),
                )
            else:
                raise SystemExit(f"Unsupported adaptive adapter: {adapter}")

            base_partitions[partition] = {
                "raw_output": str(raw),
                "scored_output": str(scored),
                "generate_argv": generate,
                "audit_raw_argv": [
                    "python",
                    "scripts/audit_run.py",
                    "--run",
                    str(raw),
                    "--stage",
                    "raw",
                    "--require-revision",
                ],
                "score_argv": [
                    "python",
                    "scripts/score_outputs.py",
                    "--run",
                    str(raw),
                    "--input",
                    refined["input"],
                    "--output",
                    str(scored),
                ],
                "audit_scored_argv": [
                    "python",
                    "scripts/audit_run.py",
                    "--run",
                    str(scored),
                    "--stage",
                    "scored",
                    "--require-revision",
                ],
            }

        result_path = (
            Path("results")
            / "adaptive"
            / adapter
            / f"{dataset}.json"
        )
        evaluate = [
            "python",
            "scripts/evaluate_adaptive.py",
            "--cal-base",
            base_partitions["calibration"]["scored_output"],
            "--cal-refined",
            cell["calibration"]["scored_output"],
            "--test-base",
            base_partitions["test"]["scored_output"],
            "--test-refined",
            cell["test"]["scored_output"],
            "--max-refine-fraction",
            str(primary_fraction),
            "--base-compute",
            str(base_compute),
            "--refined-compute",
            str(refined_compute),
            "--crossfit-folds",
            str(folds),
            "--seed",
            str(seed),
            "--n-boot",
            str(n_boot),
            "--cost-mode",
            "selective_second_pass_rerun",
            "--output",
            str(result_path),
        ]

        cells.append(
            {
                "cell": f"{adapter}:{dataset}",
                "adapter": adapter,
                "dataset": dataset,
                "base_budget": base_compute,
                "refined_budget": refined_compute,
                "execution_semantics": "selective_second_pass_rerun",
                "calibration_base": base_partitions["calibration"],
                "test_base": base_partitions["test"],
                "refined_calibration_scored": cell["calibration"]["scored_output"],
                "refined_test_scored": cell["test"]["scored_output"],
                "evaluate_argv": evaluate,
                "result": str(result_path),
            }
        )

    payload = {
        "protocol": "trajuq_adaptive_matrix_plan_v1",
        "status": "PASS",
        "primary_plan": str(primary_plan_path),
        "primary_fraction": primary_fraction,
        "execution_semantics": "selective_second_pass_rerun",
        "cell_count": len(cells),
        "cells": cells,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(out),
                "cell_count": len(cells),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
