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


def fraction_key(value: float) -> str:
    if not 0.0 <= value <= 1.0:
        raise ValueError("refinement fraction must be in [0,1]")
    return f"cap_{int(round(value * 100)):03d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plan low-budget base runs plus primary/sensitivity adaptive "
            "second-pass evaluations from a frozen primary matrix. "
            "Does not execute GPU work."
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
        "--result-root",
        default="results/adaptive",
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
    sensitivity_fractions = [
        float(value) for value in adaptive.get("sensitivity_max_refine_fraction", [])
    ]
    fraction_values = [primary_fraction, *sensitivity_fractions]
    if len(fraction_values) != len(set(fraction_values)):
        raise SystemExit("Adaptive primary/sensitivity fractions must be unique.")
    if any(not 0.0 <= value <= 1.0 for value in fraction_values):
        raise SystemExit("Adaptive refinement fractions must lie in [0,1].")

    fraction_specs = [
        {
            "key": fraction_key(value),
            "value": value,
            "primary": value == primary_fraction,
        }
        for value in fraction_values
    ]
    primary_key = fraction_key(primary_fraction)

    budgets = adaptive["budgets"]
    n_boot = int(adaptive["paired_bootstrap_n"])
    seed = int(cfg["experiment"]["seed"])
    folds = int(adaptive["crossfit_folds"])
    result_root = Path(args.result_root)

    primary_cells = primary_plan.get("cells", [])
    expected_cells = int(primary_plan.get("expected_cells", len(primary_cells)))
    if len(primary_cells) != expected_cells:
        raise SystemExit(
            f"Primary plan cell count mismatch: {len(primary_cells)} != {expected_cells}"
        )

    cells: list[dict] = []
    for cell in primary_cells:
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

        evaluations: dict[str, dict] = {}
        for fraction in fraction_specs:
            key = str(fraction["key"])
            cap = float(fraction["value"])
            cap_root = result_root / key / adapter
            result_path = cap_root / f"{dataset}.json"
            predictions_path = cap_root / f"{dataset}.predictions.jsonl"

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
                str(cap),
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
                "--predictions-output",
                str(predictions_path),
            ]
            evaluations[key] = {
                "max_refine_fraction": cap,
                "primary": bool(fraction["primary"]),
                "evaluate_argv": evaluate,
                "result": str(result_path),
                "predictions": str(predictions_path),
            }

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
                "evaluations": evaluations,
                # Backward-compatible aliases for the frozen primary cap.
                "evaluate_argv": evaluations[primary_key]["evaluate_argv"],
                "result": evaluations[primary_key]["result"],
                "predictions": evaluations[primary_key]["predictions"],
            }
        )

    macro_aggregation: dict[str, dict] = {}
    for fraction in fraction_specs:
        key = str(fraction["key"])
        macro_path = result_root / f"macro_{key}.json"
        macro_aggregation[key] = {
            "max_refine_fraction": float(fraction["value"]),
            "primary": bool(fraction["primary"]),
            "output": str(macro_path),
            "argv": [
                "python",
                "scripts/aggregate_adaptive.py",
                "--plan",
                args.output,
                "--fraction-key",
                key,
                "--n-boot",
                str(n_boot),
                "--seed",
                str(seed),
                "--output",
                str(macro_path),
            ],
        }

    payload = {
        "protocol": "trajuq_adaptive_matrix_plan_v2",
        "status": "PASS",
        "primary_plan": str(primary_plan_path),
        "primary_config": args.primary_config,
        "execution_semantics": "selective_second_pass_rerun",
        "primary_fraction": primary_fraction,
        "primary_fraction_key": primary_key,
        "sensitivity_fractions": sensitivity_fractions,
        "fractions": fraction_specs,
        "expected_cells": expected_cells,
        "cell_count": len(cells),
        "cells": cells,
        "macro_aggregation": macro_aggregation,
        "execution_order": (
            "Generate/audit/score every base-budget partition once. Then run all "
            "pre-registered cap evaluations for every cell. Only after all cells "
            "exist, execute each hierarchical macro aggregation command."
        ),
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
                "fractions": fraction_specs,
                "primary_macro": macro_aggregation[primary_key]["output"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
