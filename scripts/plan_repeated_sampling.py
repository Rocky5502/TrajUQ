#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def arg_value(argv: list[str], flag: str, default: str | None = None) -> str:
    if flag not in argv:
        if default is None:
            raise ValueError(f"Missing frozen generation flag: {flag}")
        return default
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        raise ValueError(f"Malformed argv: {flag} has no value")
    return str(argv[idx + 1])


def repeated_generation_argv(
    cell: dict,
    partition: str,
    output_dir: str,
    seeds: list[int],
    temperature: float,
    top_p: float,
) -> list[str]:
    source = cell[partition]
    frozen = list(source["generate_argv"])
    adapter = str(cell["adapter"])

    cmd = [
        "python",
        "scripts/run_repeated_samples.py",
        "--adapter",
        adapter,
        "--model",
        str(cell["model_id"]),
        "--revision",
        str(cell["model_revision"]),
        "--input",
        str(source["input"]),
        "--output-dir",
        output_dir,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--max-new-tokens",
        arg_value(frozen, "--max-new-tokens"),
        "--resume-verified",
    ]

    if adapter == "llada":
        cmd.extend(
            [
                "--steps",
                arg_value(frozen, "--steps"),
                "--block-length",
                arg_value(frozen, "--block-length"),
                "--llada-cfg-scale",
                arg_value(frozen, "--llada-cfg-scale", "0.0"),
                "--llada-remasking",
                arg_value(frozen, "--llada-remasking", "low_confidence"),
            ]
        )
    elif adapter == "dream":
        cmd.extend(
            [
                "--steps",
                arg_value(frozen, "--steps"),
                "--dream-alg",
                arg_value(frozen, "--dream-alg", "entropy"),
                "--dream-alg-temp",
                arg_value(frozen, "--dream-alg-temp", "0.1"),
            ]
        )
    elif adapter == "sdar":
        cmd.extend(
            [
                "--block-length",
                arg_value(frozen, "--block-length"),
                "--sdar-denoising-steps",
                arg_value(frozen, "--sdar-denoising-steps"),
                "--sdar-top-k",
                arg_value(frozen, "--sdar-top-k", "0"),
                "--sdar-remasking-strategy",
                arg_value(
                    frozen,
                    "--sdar-remasking-strategy",
                    "low_confidence_dynamic",
                ),
                "--sdar-confidence-threshold",
                arg_value(frozen, "--sdar-confidence-threshold", "0.85"),
            ]
        )
    else:
        raise ValueError(f"Unsupported repeated-sampling adapter: {adapter}")

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plan the frozen five-sample uncertainty baseline on the eligible "
            "model-dataset subset. This command never runs a model."
        )
    )
    parser.add_argument(
        "--primary-plan",
        default="results/primary_matrix_plan.json",
    )
    parser.add_argument(
        "--primary-analysis-plan",
        default="results/primary_analysis_plan.json",
    )
    parser.add_argument(
        "--primary-config",
        default="configs/experiments/primary.yaml",
    )
    parser.add_argument(
        "--run-root",
        default="runs/repeated",
    )
    parser.add_argument(
        "--result-root",
        default="results/repeated_sampling",
    )
    parser.add_argument(
        "--output",
        default="results/repeated_sampling_plan.json",
    )
    args = parser.parse_args()

    primary = json.loads(Path(args.primary_plan).read_text(encoding="utf-8"))
    analysis = json.loads(
        Path(args.primary_analysis_plan).read_text(encoding="utf-8")
    )
    if primary.get("status") != "PASS":
        raise SystemExit("Primary matrix plan is not PASS.")
    if analysis.get("status") != "PASS":
        raise SystemExit("Primary analysis plan is not PASS.")

    cfg = yaml.safe_load(Path(args.primary_config).read_text(encoding="utf-8"))
    repeated = cfg["baselines"]["expensive"]["repeated_sampling"]
    seeds = [int(seed) for seed in repeated["seeds"]]
    samples = int(repeated["samples"])
    if len(seeds) != samples or len(seeds) != len(set(seeds)):
        raise SystemExit("Frozen repeated-sampling seed contract is inconsistent.")

    eligible = [str(name) for name in repeated["eligible_datasets"]]
    excluded = dict(repeated.get("excluded_datasets", {}))
    if set(eligible).intersection(excluded):
        raise SystemExit("Repeated-sampling eligible/excluded datasets overlap.")

    analysis_by_cell = {
        str(item["cell"]): item
        for item in analysis.get("cell_analyses", [])
    }
    if len(analysis_by_cell) != len(analysis.get("cell_analyses", [])):
        raise SystemExit("Duplicate cells in primary analysis plan.")

    n_boot = int(cfg["statistics"]["bootstrap"]["n_boot"])
    result_root = Path(args.result_root)
    cells: list[dict] = []

    for cell in primary.get("cells", []):
        dataset = str(cell["dataset"])
        if dataset not in eligible:
            continue

        adapter = str(cell["adapter"])
        cell_name = f"{adapter}:{dataset}"
        if cell_name not in analysis_by_cell:
            raise SystemExit(f"Missing primary analysis cell: {cell_name}")

        cal_dir = Path(args.run_root) / adapter / dataset / "calibration"
        test_dir = Path(args.run_root) / adapter / dataset / "test"
        cal_cmd = repeated_generation_argv(
            cell,
            "calibration",
            str(cal_dir),
            seeds,
            float(repeated["temperature"]),
            float(repeated["top_p"]),
        )
        test_cmd = repeated_generation_argv(
            cell,
            "test",
            str(test_dir),
            seeds,
            float(repeated["temperature"]),
            float(repeated["top_p"]),
        )

        repeated_metrics = result_root / adapter / dataset / "metrics.json"
        repeated_predictions = result_root / adapter / dataset / "predictions.jsonl"
        direct_comparison = result_root / adapter / dataset / "trajuq_vs_repeated.json"

        evaluate = [
            "python",
            "scripts/evaluate_repeated_sampling.py",
            "--cal-target",
            str(cell["calibration"]["scored_output"]),
        ]
        for seed in seeds:
            evaluate.extend(
                ["--cal-sample", str(cal_dir / f"seed-{seed}.raw.jsonl")]
            )
        evaluate.extend(
            [
                "--test-target",
                str(cell["test"]["scored_output"]),
            ]
        )
        for seed in seeds:
            evaluate.extend(
                ["--test-sample", str(test_dir / f"seed-{seed}.raw.jsonl")]
            )
        evaluate.extend(
            [
                "--expected-samples",
                str(samples),
                "--seed",
                str(cfg["experiment"]["seed"]),
                "--output",
                str(repeated_metrics),
                "--predictions-output",
                str(repeated_predictions),
            ]
        )

        trajuq_predictions = str(
            analysis_by_cell[cell_name]["outputs"]["predictions"]
        )
        compare = [
            "python",
            "scripts/compare_repeated_to_trajuq.py",
            "--trajuq-predictions",
            trajuq_predictions,
            "--repeated-predictions",
            str(repeated_predictions),
            "--n-boot",
            str(n_boot),
            "--seed",
            str(cfg["experiment"]["seed"]),
            "--output",
            str(direct_comparison),
        ]

        cells.append(
            {
                "cell": cell_name,
                "adapter": adapter,
                "dataset": dataset,
                "sample_count": samples,
                "seeds": seeds,
                "calibration_generation_argv": cal_cmd,
                "test_generation_argv": test_cmd,
                "evaluate_argv": evaluate,
                "compare_to_trajuq_argv": compare,
                "trajuq_predictions": trajuq_predictions,
                "repeated_predictions": str(repeated_predictions),
                "repeated_metrics": str(repeated_metrics),
                "direct_comparison": str(direct_comparison),
            }
        )

    expected_cells = len(cfg["models"]["primary"]) * len(eligible)
    if len(cells) != expected_cells:
        raise SystemExit(
            f"Repeated baseline cell count mismatch: {len(cells)} != {expected_cells}"
        )

    macro_path = result_root / "macro_trajuq_vs_repeated.json"
    macro_argv = [
        "python",
        "scripts/aggregate_repeated_baseline.py",
        "--plan",
        args.output,
        "--n-boot",
        str(n_boot),
        "--seed",
        str(cfg["experiment"]["seed"]),
        "--output",
        str(macro_path),
    ]

    payload = {
        "protocol": "trajuq_repeated_sampling_plan_v1",
        "status": "PASS",
        "primary_plan": args.primary_plan,
        "primary_analysis_plan": args.primary_analysis_plan,
        "primary_config": args.primary_config,
        "samples": samples,
        "seeds": seeds,
        "temperature": float(repeated["temperature"]),
        "top_p": float(repeated["top_p"]),
        "eligible_datasets": eligible,
        "excluded_datasets": excluded,
        "expected_cells": expected_cells,
        "cell_count": len(cells),
        "cells": cells,
        "macro_aggregation_argv": macro_argv,
        "macro_output": str(macro_path),
        "execution_order": (
            "Generate all calibration/test auxiliary samples, evaluate every "
            "eligible cell, run paired per-cell comparisons, then macro-aggregate "
            "the complete matched subset."
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
                "cell_count": len(cells),
                "macro_output": str(macro_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
