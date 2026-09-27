#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.execute_primary_plan import (  # noqa: E402
    validate_plan_provenance as validate_primary_plan_provenance,
)
from scripts.plan_primary_analysis import scored_source_chain_passes  # noqa: E402


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

    failures: list[str] = []
    primary_plan_path = Path(args.primary_plan)
    cfg_path = Path(args.primary_config)
    if not primary_plan_path.exists():
        failures.append(f"missing primary matrix plan: {primary_plan_path}")
    if not cfg_path.exists():
        failures.append(f"missing primary config: {cfg_path}")
    if failures:
        raise SystemExit("\n".join(failures))

    primary_plan = json.loads(
        primary_plan_path.read_text(encoding="utf-8")
    )
    if primary_plan.get("protocol") != "trajuq_primary_matrix_plan_v2":
        failures.append("primary matrix plan protocol is not v2")
    if primary_plan.get("status") != "PASS":
        failures.append("primary matrix plan is not PASS")
    failures.extend(
        "primary plan provenance: " + failure
        for failure in validate_primary_plan_provenance(primary_plan)
    )

    experiment_commit = str(
        primary_plan.get("experiment_git_commit", "")
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
            "current git worktree is dirty; adaptive planning requires the "
            "frozen clean checkout"
        )

    cfg_hash = sha256_file(cfg_path)
    if str(primary_plan.get("primary_config_sha256", "")) != cfg_hash:
        failures.append(
            "primary config hash differs from frozen primary matrix plan"
        )

    if failures:
        payload = {
            "protocol": "trajuq_adaptive_matrix_plan_v3",
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

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    adaptive = cfg["adaptive_compute"]
    if adaptive.get("execution_semantics") != "selective_second_pass_rerun":
        raise SystemExit(
            "Primary adaptive execution semantics changed unexpectedly."
        )

    primary_fraction = float(adaptive["primary_max_refine_fraction"])
    sensitivity_fractions = [
        float(value)
        for value in adaptive.get(
            "sensitivity_max_refine_fraction",
            [],
        )
    ]
    fraction_values = [primary_fraction, *sensitivity_fractions]
    if len(fraction_values) != len(set(fraction_values)):
        raise SystemExit(
            "Adaptive primary/sensitivity fractions must be unique."
        )
    if any(not 0.0 <= value <= 1.0 for value in fraction_values):
        raise SystemExit(
            "Adaptive refinement fractions must lie in [0,1]."
        )

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
    expected_cells = int(
        primary_plan.get("expected_cells", len(primary_cells))
    )
    if len(primary_cells) != expected_cells:
        raise SystemExit(
            f"Primary plan cell count mismatch: {len(primary_cells)} != "
            f"{expected_cells}"
        )

    cells: list[dict] = []
    for cell in primary_cells:
        adapter = str(cell["adapter"])
        dataset = str(cell["dataset"])
        cell_label = f"{adapter}:{dataset}"
        if adapter not in budgets:
            raise SystemExit(f"No frozen adaptive budget for {adapter}")
        budget = budgets[adapter]

        provenance: dict[str, dict[str, str]] = {}
        base_partitions: dict[str, dict] = {}
        for partition in ("calibration", "test"):
            refined = cell[partition]
            generation_input = Path(str(refined["input"]))
            expected_input_hash = str(
                refined.get("input_sha256", "")
            )
            if (
                not generation_input.exists()
                or not expected_input_hash
                or sha256_file(generation_input) != expected_input_hash
            ):
                raise SystemExit(
                    f"{cell_label} {partition}: frozen input drifted"
                )

            raw_refined = Path(str(refined["raw_output"]))
            refined_scored = Path(str(refined["scored_output"]))
            if not scored_source_chain_passes(
                generation_input,
                raw_refined,
                refined_scored,
                experiment_commit,
            ):
                raise SystemExit(
                    f"{cell_label} {partition}: refined scored artifact does "
                    "not trace to the frozen input and clean raw audit"
                )
            if not scored_audit_passes(refined_scored, experiment_commit):
                raise SystemExit(
                    f"{cell_label} {partition}: refined scored artifact is "
                    "missing, invalid, or stale"
                )
            refined_audit = refined_scored.with_suffix(
                refined_scored.suffix + ".audit.json"
            )
            provenance[partition] = {
                "generation_input": str(generation_input),
                "generation_input_sha256": sha256_file(
                    generation_input
                ),
                "refined_scored": str(refined_scored),
                "refined_scored_sha256": sha256_file(refined_scored),
                "refined_audit": str(refined_audit),
                "refined_audit_sha256": sha256_file(refined_audit),
            }

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
                base_compute = float(
                    budget["base_denoising_steps"]
                )
                refined_compute = float(
                    budget["refined_denoising_steps"]
                )
                generate = replace_arg(
                    generate,
                    "--sdar-denoising-steps",
                    str(int(budget["base_denoising_steps"])),
                )
            else:
                raise SystemExit(
                    f"Unsupported adaptive adapter: {adapter}"
                )

            base_partitions[partition] = {
                "input": str(generation_input),
                "input_sha256": sha256_file(generation_input),
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
                    str(generation_input),
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
            predictions_path = (
                cap_root / f"{dataset}.predictions.jsonl"
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
                "cell": cell_label,
                "adapter": adapter,
                "dataset": dataset,
                "base_budget": base_compute,
                "refined_budget": refined_compute,
                "execution_semantics": "selective_second_pass_rerun",
                "provenance": provenance,
                "calibration_base": base_partitions["calibration"],
                "test_base": base_partitions["test"],
                "refined_calibration_scored": (
                    cell["calibration"]["scored_output"]
                ),
                "refined_test_scored": cell["test"]["scored_output"],
                "evaluations": evaluations,
                "evaluate_argv": (
                    evaluations[primary_key]["evaluate_argv"]
                ),
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
        "protocol": "trajuq_adaptive_matrix_plan_v3",
        "status": "PASS",
        "git_commit": current_head,
        "experiment_git_commit": experiment_commit,
        "primary_plan": str(primary_plan_path),
        "primary_plan_sha256": sha256_file(primary_plan_path),
        "primary_config": str(cfg_path),
        "primary_config_sha256": cfg_hash,
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
            "Generate/audit/score every base-budget partition once. Then run "
            "all pre-registered cap evaluations for every cell. Only after all "
            "cells exist, execute each hierarchical macro aggregation command."
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
                "primary_macro": (
                    macro_aggregation[primary_key]["output"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
