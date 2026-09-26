#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from scripts.plan_primary_analysis import scored_source_chain_passes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def git_status_porcelain() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def scored_audit_passes(
    path: Path,
    expected_commit: str | None,
) -> bool:
    audit = path.with_suffix(path.suffix + ".audit.json")
    if not path.exists() or not audit.exists():
        return False
    try:
        payload = json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("protocol") == "trajuq_run_audit_v1"
        and payload.get("status") == "PASS"
        and payload.get("git_clean") is True
        and payload.get("git_commit") == expected_commit
        and payload.get("stage") == "scored"
        and str(payload.get("sha256", "")) == sha256_file(path)
    )


def valid_jsonl(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    return bool(rows)


def jsonl_example_ids(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSONL artifact {path}: {exc}") from exc
    ids = [str(row.get("example_id", "")) for row in rows]
    if not rows or any(not item_id for item_id in ids):
        raise ValueError(f"invalid example IDs in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate example IDs in {path}")
    return ids


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
            "Plan the frozen five-sample uncertainty baseline on the complete "
            "primary model-dataset matrix. This command never runs a model."
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

    failures: list[str] = []
    primary_path = Path(args.primary_plan)
    analysis_path = Path(args.primary_analysis_plan)
    cfg_path = Path(args.primary_config)
    for path in (primary_path, analysis_path, cfg_path):
        if not path.exists():
            failures.append(f"missing required artifact: {path}")
    if failures:
        raise SystemExit("\n".join(failures))

    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if primary.get("protocol") != "trajuq_primary_matrix_plan_v2":
        failures.append("primary matrix plan protocol is not v2")
    if primary.get("status") != "PASS":
        failures.append("primary matrix plan is not PASS")
    if analysis.get("protocol") != "trajuq_primary_analysis_plan_v2":
        failures.append("primary analysis plan protocol is not v2")
    if analysis.get("status") != "PASS":
        failures.append("primary analysis plan is not PASS")

    experiment_commit = str(primary.get("experiment_git_commit", ""))
    if str(analysis.get("experiment_git_commit", "")) != experiment_commit:
        failures.append(
            "primary and analysis plans disagree on the experiment git commit"
        )
    current_head = git_head()
    if not current_head:
        failures.append("current git commit could not be resolved")
    elif current_head != experiment_commit:
        failures.append(
            "current git commit differs from the frozen experiment commit"
        )
    current_status = git_status_porcelain()
    if current_status is None:
        failures.append("current git worktree status could not be resolved")
    elif current_status:
        failures.append(
            "current git worktree is dirty; repeated-sampling planning requires "
            "the frozen clean checkout"
        )

    current_cfg_hash = sha256_file(cfg_path)
    if str(primary.get("primary_config_sha256", "")) != current_cfg_hash:
        failures.append("primary config hash differs from primary matrix plan")
    if str(analysis.get("primary_config_sha256", "")) != current_cfg_hash:
        failures.append("primary config hash differs from primary analysis plan")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    repeated = cfg["baselines"]["expensive"]["repeated_sampling"]
    seeds = [int(seed) for seed in repeated["seeds"]]
    samples = int(repeated["samples"])
    if len(seeds) != samples or len(seeds) != len(set(seeds)):
        failures.append("frozen repeated-sampling seed contract is inconsistent")

    eligible = [str(name) for name in repeated["eligible_datasets"]]
    excluded = dict(repeated.get("excluded_datasets", {}))
    if set(eligible).intersection(excluded):
        failures.append("repeated-sampling eligible/excluded datasets overlap")

    analysis_by_cell = {
        str(item["cell"]): item
        for item in analysis.get("cell_analyses", [])
    }
    if len(analysis_by_cell) != len(analysis.get("cell_analyses", [])):
        failures.append("duplicate cells in primary analysis plan")

    if failures:
        payload = {
            "protocol": "trajuq_repeated_sampling_plan_v2",
            "status": "FAIL",
            "experiment_git_commit": experiment_commit or None,
            "failures": failures,
            "cells": [],
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(1)

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

        provenance: dict[str, dict[str, str]] = {}
        for partition in ("calibration", "test"):
            source = cell[partition]
            generation_input = Path(str(source["input"]))
            expected_input_hash = str(source.get("input_sha256", ""))
            if (
                not generation_input.exists()
                or not expected_input_hash
                or sha256_file(generation_input) != expected_input_hash
            ):
                raise SystemExit(
                    f"{cell_name} {partition}: frozen generation input drifted"
                )

            raw = Path(str(source["raw_output"]))
            scored = Path(str(source["scored_output"]))
            if not scored_source_chain_passes(
                generation_input,
                raw,
                scored,
                experiment_commit,
            ):
                raise SystemExit(
                    f"{cell_name} {partition}: scored target does not trace "
                    "to the frozen input and clean raw audit"
                )
            if not scored_audit_passes(scored, experiment_commit):
                raise SystemExit(
                    f"{cell_name} {partition}: scored target is missing, "
                    "invalid, or stale"
                )
            scored_audit = scored.with_suffix(scored.suffix + ".audit.json")
            provenance[partition] = {
                "generation_input": str(generation_input),
                "generation_input_sha256": sha256_file(generation_input),
                "scored_target": str(scored),
                "scored_target_sha256": sha256_file(scored),
                "scored_audit": str(scored_audit),
                "scored_audit_sha256": sha256_file(scored_audit),
            }

        trajuq_predictions = Path(
            str(analysis_by_cell[cell_name]["outputs"]["predictions"])
        )
        if not valid_jsonl(trajuq_predictions):
            raise SystemExit(
                f"{cell_name}: primary TrajUQ predictions are missing or invalid"
            )
        try:
            prediction_ids = jsonl_example_ids(trajuq_predictions)
            frozen_test_ids = jsonl_example_ids(
                Path(str(cell["test"]["scored_output"]))
            )
        except ValueError as exc:
            raise SystemExit(f"{cell_name}: {exc}") from exc
        if prediction_ids != frozen_test_ids:
            raise SystemExit(
                f"{cell_name}: primary TrajUQ predictions do not exactly "
                "cover the frozen scored test target"
            )

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
        repeated_predictions = (
            result_root / adapter / dataset / "predictions.jsonl"
        )
        direct_comparison = (
            result_root / adapter / dataset / "trajuq_vs_repeated.json"
        )

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

        compare = [
            "python",
            "scripts/compare_repeated_to_trajuq.py",
            "--trajuq-predictions",
            str(trajuq_predictions),
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
                "provenance": {
                    **provenance,
                    "trajuq_predictions": str(trajuq_predictions),
                    "trajuq_predictions_sha256": sha256_file(
                        trajuq_predictions
                    ),
                },
                "calibration_generation_argv": cal_cmd,
                "test_generation_argv": test_cmd,
                "evaluate_argv": evaluate,
                "compare_to_trajuq_argv": compare,
                "trajuq_predictions": str(trajuq_predictions),
                "repeated_predictions": str(repeated_predictions),
                "repeated_metrics": str(repeated_metrics),
                "direct_comparison": str(direct_comparison),
            }
        )

    expected_cells = len(cfg["models"]["primary"]) * len(eligible)
    if len(cells) != expected_cells:
        raise SystemExit(
            f"Repeated baseline cell count mismatch: {len(cells)} != "
            f"{expected_cells}"
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
        "protocol": "trajuq_repeated_sampling_plan_v2",
        "status": "PASS",
        "git_commit": current_head,
        "experiment_git_commit": experiment_commit,
        "primary_plan": str(primary_path),
        "primary_plan_sha256": sha256_file(primary_path),
        "primary_analysis_plan": str(analysis_path),
        "primary_analysis_plan_sha256": sha256_file(analysis_path),
        "primary_config": str(cfg_path),
        "primary_config_sha256": current_cfg_hash,
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
            "cell, run paired per-cell comparisons, then macro-aggregate the "
            "complete 18-cell matrix."
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
                "macro_output": str(macro_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
