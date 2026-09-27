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
from trajuq.io import read_jsonl, write_json_immutable  # noqa: E402


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


def audit_path(artifact: Path) -> Path:
    return artifact.with_suffix(artifact.suffix + ".audit.json")


def exact_id_coverage(input_path: Path, artifact_path: Path) -> bool:
    try:
        input_rows = list(read_jsonl(input_path))
        artifact_rows = list(read_jsonl(artifact_path))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return False

    input_ids = [str(row.get("id", "")) for row in input_rows]
    artifact_ids = [str(row.get("example_id", "")) for row in artifact_rows]
    if (
        not input_rows
        or any(not item_id for item_id in input_ids)
        or any(not item_id for item_id in artifact_ids)
        or len(input_ids) != len(set(input_ids))
        or len(artifact_ids) != len(set(artifact_ids))
    ):
        return False
    return artifact_ids == input_ids


def run_audit_passes(
    artifact: Path,
    *,
    stage: str,
    expected_commit: str | None,
) -> bool:
    audit = audit_path(artifact)
    if not artifact.exists() or not audit.exists():
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
        and payload.get("stage") == stage
        and str(payload.get("sha256", "")) == sha256_file(artifact)
    )


def scored_source_chain_passes(
    source_input: Path,
    raw: Path,
    scored: Path,
    expected_commit: str | None,
) -> bool:
    if not exact_id_coverage(source_input, raw):
        return False
    if not exact_id_coverage(source_input, scored):
        return False
    if not run_audit_passes(
        raw,
        stage="raw",
        expected_commit=expected_commit,
    ):
        return False

    raw_audit = audit_path(raw)
    expected = {
        "raw_run_sha256": sha256_file(raw),
        "raw_audit_sha256": sha256_file(raw_audit),
        "raw_audit_git_commit": expected_commit,
        "scoring_input_sha256": sha256_file(source_input),
    }
    try:
        rows = list(read_jsonl(scored))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return False
    if not rows:
        return False

    for row in rows:
        metadata = row.get("metadata", {})
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        if not metadata.get("scorer"):
            return False
    return True


def audit_passes(
    artifact: Path,
    expected_commit: str | None,
) -> bool:
    return run_audit_passes(
        artifact,
        stage="scored",
        expected_commit=expected_commit,
    )


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

    failures: list[str] = []
    plan_path = Path(args.primary_plan)
    if not plan_path.exists():
        raise SystemExit(f"Missing primary plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("protocol") != "trajuq_primary_matrix_plan_v2":
        failures.append("primary matrix plan protocol is not v2")
    if plan.get("status") != "PASS":
        failures.append("primary matrix plan is not PASS")
    failures.extend(
        "primary plan provenance: " + failure
        for failure in validate_primary_plan_provenance(plan)
    )

    experiment_commit = str(plan.get("experiment_git_commit", ""))
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
            "current git worktree is dirty; analysis planning requires the "
            "frozen clean checkout"
        )

    cfg_path = Path(args.primary_config)
    if not cfg_path.exists():
        failures.append(f"missing primary config: {cfg_path}")
        cfg: dict = {}
    else:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        expected_cfg_hash = str(plan.get("primary_config_sha256", ""))
        if not expected_cfg_hash:
            failures.append("primary matrix plan has no primary-config hash")
        elif sha256_file(cfg_path) != expected_cfg_hash:
            failures.append(
                "primary config hash differs from the frozen primary matrix plan"
            )

    expected_cells = int(plan.get("expected_cells", 0))
    cells = plan.get("cells", [])
    if expected_cells <= 0 or len(cells) != expected_cells:
        failures.append(
            f"primary plan cell count mismatch: {len(cells)} != {expected_cells}"
        )

    seen_pairs: set[tuple[str, str]] = set()
    input_bindings: dict[str, dict[str, dict[str, str]]] = {}
    for cell in cells:
        adapter = str(cell["adapter"])
        dataset = str(cell["dataset"])
        pair = (adapter, dataset)
        if pair in seen_pairs:
            failures.append(f"duplicate primary cell: {adapter}:{dataset}")
            continue
        seen_pairs.add(pair)

        partitions: dict[str, dict[str, str]] = {}
        for partition in ("calibration", "test"):
            source_input = Path(str(cell[partition]["input"]))
            raw = Path(str(cell[partition]["raw_output"]))
            scored = Path(str(cell[partition]["scored_output"]))
            if not raw.exists():
                failures.append(
                    f"{adapter}:{dataset} {partition}: missing raw artifact {raw}"
                )
                continue
            if not scored.exists():
                failures.append(
                    f"{adapter}:{dataset} {partition}: missing scored artifact {scored}"
                )
                continue
            if not scored_source_chain_passes(
                source_input,
                raw,
                scored,
                experiment_commit,
            ):
                failures.append(
                    f"{adapter}:{dataset} {partition}: raw/scored source chain "
                    "is incomplete, stale, or not hash-bound"
                )
                continue
            if not audit_passes(scored, experiment_commit):
                failures.append(
                    f"{adapter}:{dataset} {partition}: scored audit is not PASS "
                    "or is stale"
                )
                continue
            partitions[partition] = {
                "source_input": str(source_input),
                "source_input_sha256": sha256_file(source_input),
                "raw_path": str(raw),
                "raw_sha256": sha256_file(raw),
                "raw_audit_path": str(audit_path(raw)),
                "raw_audit_sha256": sha256_file(audit_path(raw)),
                "path": str(scored),
                "sha256": sha256_file(scored),
                "audit_path": str(audit_path(scored)),
                "audit_sha256": sha256_file(audit_path(scored)),
            }
        input_bindings[f"{adapter}:{dataset}"] = partitions

    if failures:
        payload = {
            "protocol": "trajuq_primary_analysis_plan_v2",
            "status": "FAIL",
            "experiment_git_commit": experiment_commit or None,
            "failures": failures,
            "cell_analyses": [],
        }
        out = Path(args.output)
        write_json_immutable(out, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(1)

    n_boot = int(cfg["statistics"]["bootstrap"]["n_boot"])
    seed = int(cfg["experiment"]["seed"])
    sensitivity_methods = list(cfg.get("calibration", {}).get("sensitivity", []))
    if sensitivity_methods != ["isotonic"]:
        raise SystemExit(
            "Frozen calibration sensitivity must be exactly ['isotonic']."
        )
    isotonic_cfg = dict(cfg.get("calibration", {}).get("isotonic", {}))
    required_isotonic = {
        "role": "supplementary_calibration_sensitivity",
        "base_estimator": "standardized_logistic",
        "cv_folds": int(isotonic_cfg.get("cv_folds", 0)),
        "ensemble": bool(isotonic_cfg.get("ensemble", True)),
        "min_calibration_samples": int(
            isotonic_cfg.get("min_calibration_samples", 0)
        ),
        "min_class_count": int(isotonic_cfg.get("min_class_count", 0)),
    }
    if (
        required_isotonic["cv_folds"] != 5
        or required_isotonic["ensemble"] is not False
        or required_isotonic["min_calibration_samples"] != 1000
        or required_isotonic["min_class_count"] != 5
    ):
        raise SystemExit("Unexpected frozen isotonic sensitivity contract.")
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
        "--analysis-plan",
        args.output,
        "--output",
        str(root / "macro_primary.json"),
    ]

    for cell in cells:
        adapter = str(cell["adapter"])
        dataset = str(cell["dataset"])
        cell_label = f"{adapter}:{dataset}"
        cal = str(cell["calibration"]["scored_output"])
        test = str(cell["test"]["scored_output"])
        cell_root = root / adapter / dataset
        metrics = cell_root / "metrics.json"
        predictions = cell_root / "predictions.jsonl"
        nested = cell_root / "nested_bootstrap.json"
        ablations = cell_root / "feature_ablation.json"
        ablation_predictions = cell_root / "feature_ablation_predictions.jsonl"
        prior_diag = cell_root / "prior_signal_diagnostic.json"
        prior_predictions = cell_root / "prior_signal_predictions.jsonl"
        calibration_sensitivity = cell_root / "calibration_sensitivity.json"
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
            "--predictions-output",
            str(ablation_predictions),
        ]
        prior_signal_diagnostic = [
            "python",
            "scripts/evaluate_prior_signal_diagnostics.py",
            "--calibration",
            cal,
            "--test",
            test,
            "--seed",
            str(seed),
            "--output",
            str(prior_diag),
            "--predictions-output",
            str(prior_predictions),
        ]
        calibration_sensitivity_argv = [
            "python",
            "scripts/evaluate_calibration_sensitivity.py",
            "--calibration",
            cal,
            "--test",
            test,
            "--primary-config",
            args.primary_config,
            "--seed",
            str(seed),
            "--output",
            str(calibration_sensitivity),
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
                "cell": cell_label,
                "adapter": adapter,
                "dataset": dataset,
                "experiment_git_commit": experiment_commit,
                "inputs": input_bindings[cell_label],
                "evaluate_argv": evaluate,
                "nested_comparison_argv": compare_nested,
                "feature_ablation_argv": ablation,
                "prior_signal_diagnostic_argv": prior_signal_diagnostic,
                "calibration_sensitivity_argv": calibration_sensitivity_argv,
                "calibration_sensitivity_contract": required_isotonic,
                "raw_risk_diagnostic_argv": diagnostic,
                "outputs": {
                    "metrics": str(metrics),
                    "predictions": str(predictions),
                    "nested_bootstrap": str(nested),
                    "feature_ablation": str(ablations),
                    "feature_ablation_predictions": str(ablation_predictions),
                    "prior_signal_diagnostic": str(prior_diag),
                    "prior_signal_predictions": str(prior_predictions),
                    "calibration_sensitivity": str(calibration_sensitivity),
                    "raw_risk_diagnostic": str(raw_diag),
                },
            }
        )

    macro_path = root / "macro_primary.json"
    ablation_macro_path = root / "macro_feature_ablation.json"
    prior_signal_macro_path = root / "macro_prior_signal_diagnostics.json"
    gate_path = root / "rq1_claim_gate.json"
    ablation_macro_argv = [
        "python",
        "scripts/aggregate_feature_ablation.py",
        "--plan",
        args.output,
        "--n-boot",
        str(n_boot),
        "--seed",
        str(seed),
        "--output",
        str(ablation_macro_path),
    ]
    prior_signal_macro_argv = [
        "python",
        "scripts/aggregate_prior_signal_diagnostics.py",
        "--plan",
        args.output,
        "--n-boot",
        str(n_boot),
        "--seed",
        str(seed),
        "--output",
        str(prior_signal_macro_path),
    ]
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
        "protocol": "trajuq_primary_analysis_plan_v2",
        "status": "PASS",
        "git_commit": current_head,
        "experiment_git_commit": experiment_commit,
        "primary_plan": str(plan_path),
        "primary_plan_sha256": sha256_file(plan_path),
        "primary_config": str(cfg_path),
        "primary_config_sha256": sha256_file(cfg_path),
        "seed": seed,
        "n_boot": n_boot,
        "cell_count": len(analyses),
        "cell_analyses": analyses,
        "macro_aggregation_argv": aggregation_args,
        "feature_ablation_macro_argv": ablation_macro_argv,
        "prior_signal_macro_argv": prior_signal_macro_argv,
        "rq1_claim_gate_argv": gate_argv,
        "final_outputs": {
            "macro_primary": str(macro_path),
            "macro_feature_ablation": str(ablation_macro_path),
            "macro_prior_signal_diagnostics": str(prior_signal_macro_path),
            "rq1_claim_gate": str(gate_path),
        },
        "execution_order": (
            "Run all cell analyses, including the frozen supplementary "
            "isotonic calibration sensitivity (or an explicit eligibility skip), "
            "then primary macro aggregation, then the complete feature-ablation "
            "and prior-aligned signal macros, "
            "then the RQ1 claim gate. Do "
            "not omit a weak cell or rerun with altered analysis settings."
        ),
    }

    out = Path(args.output)
    write_json_immutable(out, payload)
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
