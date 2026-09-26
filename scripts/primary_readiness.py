#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


DEFAULT_PARITY = {
    "llada": "results/llada_parity.json",
    "sdar": "results/sdar_parity.json",
    "dream": "results/dream_parity.json",
}


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_override(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--parity must use MODEL=PATH")
    model, path = value.split("=", 1)
    model, path = model.strip(), path.strip()
    if not model or not path:
        raise argparse.ArgumentTypeError("--parity MODEL and PATH must be non-empty")
    return model, path


def evaluate_readiness(
    primary_config: str,
    pilot_quality_path: str,
    manual_gate_path: str,
    parity_paths: dict[str, str],
    math500_audit_path: str = "results/math500_scorer_audit.json",
    math500_manual_path: str = "results/math500_scorer_manual_review.json",
    cpu_verification_path: str = "results/cpu_verification.json",
    revision_ledger_path: str = "results/revision_ledger.json",
) -> dict:
    failures: list[str] = []
    checks: dict[str, dict] = {}

    cfg_path = Path(primary_config)
    primary_models: list[str] = []
    primary_datasets: list[str] = []
    cfg: dict = {}
    if not cfg_path.exists():
        failures.append(f"missing primary config: {primary_config}")
    else:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        primary_models = [
            Path(path).stem
            for path in cfg.get("models", {}).get("primary", [])
        ]
        primary_datasets = [str(name) for name in cfg.get("datasets", [])]
        if not primary_models:
            failures.append("primary config contains no primary models")
        if not primary_datasets:
            failures.append("primary config contains no primary datasets")

    experiment_commit: str | None = None
    ledger_models: dict[str, dict] = {}
    try:
        ledger = load_json(revision_ledger_path)
        experiment_commit = ledger.get("git_commit")
        ledger_models = {
            str(item.get("adapter")): item
            for item in ledger.get("models", [])
            if item.get("adapter")
        }
        ledger_datasets = {
            str(item.get("name")): item
            for item in ledger.get("datasets", [])
            if item.get("name")
        }
        ledger_pass = (
            ledger.get("protocol") == "trajuq_revision_ledger_v2"
            and ledger.get("status") == "PASS"
            and isinstance(experiment_commit, str)
            and bool(experiment_commit)
            and set(ledger_models) == set(primary_models)
            and set(ledger_datasets) == set(primary_datasets)
        )

        observed_primary_sha = sha256_file(cfg_path) if cfg_path.exists() else None
        if observed_primary_sha != ledger.get("primary_config_sha256"):
            ledger_pass = False
            failures.append("primary config hash differs from revision ledger")

        frozen_configs = ledger.get("frozen_model_configs", {})
        frozen_hashes = ledger.get("frozen_model_config_sha256", {})
        for model in primary_models:
            frozen_path = Path(str(frozen_configs.get(model, "")))
            expected_hash = frozen_hashes.get(model)
            if not frozen_path.exists() or not expected_hash:
                ledger_pass = False
                failures.append(f"{model}: frozen model config missing from revision ledger")
                continue
            if sha256_file(frozen_path) != expected_hash:
                ledger_pass = False
                failures.append(f"{model}: frozen model config hash differs from revision ledger")
            try:
                frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
                if str(frozen.get("revision")) != str(
                    ledger_models.get(model, {}).get("resolved_sha")
                ):
                    ledger_pass = False
                    failures.append(
                        f"{model}: frozen config revision differs from revision ledger"
                    )
            except (OSError, yaml.YAMLError, TypeError) as exc:
                ledger_pass = False
                failures.append(f"{model}: frozen config invalid: {exc}")

        checks["revision_ledger"] = {
            "path": revision_ledger_path,
            "status": ledger.get("status"),
            "protocol": ledger.get("protocol"),
            "git_commit": experiment_commit,
            "primary_config_sha256": ledger.get("primary_config_sha256"),
            "model_adapters": sorted(ledger_models),
            "datasets": sorted(ledger_datasets),
        }
        if not ledger_pass:
            failures.append("immutable revision ledger is not a valid frozen PASS")
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as exc:
        failures.append(f"revision ledger unavailable/invalid: {exc}")

    try:
        cpu = load_json(cpu_verification_path)
        cpu_steps = cpu.get("steps", [])
        cpu_pass = (
            cpu.get("status") == "PASS"
            and cpu.get("git_clean") is True
            and len(cpu_steps) == len(cpu.get("expected_steps", []))
            and all(step.get("status") == "PASS" for step in cpu_steps)
            and experiment_commit is not None
            and cpu.get("git_commit") == experiment_commit
        )
        checks["cpu_verification"] = {
            "path": cpu_verification_path,
            "status": cpu.get("status"),
            "git_commit": cpu.get("git_commit"),
            "git_clean": cpu.get("git_clean"),
            "completed_steps": cpu.get("completed_steps"),
        }
        if not cpu_pass:
            failures.append(
                "CPU repository verification is not a clean PASS at the frozen experiment commit"
            )
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as exc:
        failures.append(f"CPU verification artifact unavailable/invalid: {exc}")

    try:
        quality = load_json(pilot_quality_path)
        expected_llada_revision = ledger_models.get("llada", {}).get("resolved_sha")
        raw_path = Path(str(quality.get("raw", "")))
        scored_path = Path(str(quality.get("scored", "")))
        raw_hash_ok = (
            raw_path.exists()
            and quality.get("raw_sha256") == sha256_file(raw_path)
        )
        scored_hash_ok = (
            scored_path.exists()
            and quality.get("scored_sha256") == sha256_file(scored_path)
        )
        quality_pass = (
            quality.get("protocol") == "trajuq_pilot_quality_gate_v2"
            and quality.get("status") == "PASS"
            and quality.get("git_commit") == experiment_commit
            and int(quality.get("raw_records", 0)) == 50
            and int(quality.get("scored_records", 0)) == 50
            and quality.get("adapters") == ["llada"]
            and quality.get("model_revisions") == [expected_llada_revision]
            and raw_hash_ok
            and scored_hash_ok
        )
        checks["pilot_quality"] = {
            "path": pilot_quality_path,
            "status": quality.get("status"),
            "protocol": quality.get("protocol"),
            "git_commit": quality.get("git_commit"),
            "records": quality.get("raw_records"),
            "scored_records": quality.get("scored_records"),
            "model_revisions": quality.get("model_revisions"),
            "raw_hash_ok": raw_hash_ok,
            "scored_hash_ok": scored_hash_ok,
        }
        if not quality_pass:
            failures.append(
                "50-example pilot quality gate is not a hash-bound PASS at the frozen commit/revision"
            )
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as exc:
        failures.append(f"pilot quality artifact unavailable/invalid: {exc}")

    try:
        manual = load_json(manual_gate_path)
        manual_checks = manual.get("checks", {})
        review_sheet = Path(str(manual.get("review_sheet", "")))
        llada_parity_path = Path(str(parity_paths.get("llada", "")))
        quality_artifact = Path(pilot_quality_path)
        review_hash_ok = (
            review_sheet.exists()
            and manual.get("review_sheet_sha256") == sha256_file(review_sheet)
        )
        parity_hash_ok = (
            llada_parity_path.exists()
            and manual.get("parity_artifact_sha256")
            == sha256_file(llada_parity_path)
        )
        quality_hash_ok = (
            quality_artifact.exists()
            and manual.get("pilot_quality_sha256")
            == sha256_file(quality_artifact)
        )
        required_manual = (
            manual.get("protocol") == "trajuq_pilot_manual_gate_v2"
            and manual.get("status") == "PASS"
            and manual.get("git_commit") == experiment_commit
            and int(manual.get("reviewed_records", 0)) >= 10
            and review_hash_ok
            and parity_hash_ok
            and quality_hash_ok
            and all(
                manual_checks.get(name) is True
                for name in (
                    "trajectory_semantics",
                    "scorer_audit",
                    "parity_artifact_reviewed",
                    "no_leakage_observed",
                )
            )
        )
        checks["pilot_manual"] = {
            "path": manual_gate_path,
            "status": manual.get("status"),
            "protocol": manual.get("protocol"),
            "git_commit": manual.get("git_commit"),
            "reviewed_records": manual.get("reviewed_records"),
            "review_sheet_hash_ok": review_hash_ok,
            "parity_hash_ok": parity_hash_ok,
            "pilot_quality_hash_ok": quality_hash_ok,
        }
        if not required_manual:
            failures.append("human pilot-review gate is incomplete or stale")
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as exc:
        failures.append(f"manual pilot artifact unavailable/invalid: {exc}")

    try:
        math_audit = load_json(math500_audit_path)
        math_audit_pass = (
            math_audit.get("protocol") == "trajuq_math500_scorer_audit_v2"
            and math_audit.get("status") == "PASS"
            and math_audit.get("git_commit") == experiment_commit
            and int(math_audit.get("failure_count", 1)) == 0
        )
        checks["math500_scorer"] = {
            "path": math500_audit_path,
            "status": math_audit.get("status"),
            "protocol": math_audit.get("protocol"),
            "git_commit": math_audit.get("git_commit"),
            "failure_count": math_audit.get("failure_count"),
        }
        if not math_audit_pass:
            failures.append(
                "MATH-500 automatic scorer audit is not a PASS at the frozen experiment commit"
            )
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as exc:
        failures.append(f"MATH-500 scorer audit unavailable/invalid: {exc}")

    try:
        math_manual = load_json(math500_manual_path)
        math_checks = math_manual.get("checks", {})
        math_review_sheet = Path(str(math_manual.get("review_sheet", "")))
        math_audit_artifact = Path(math500_audit_path)
        math_review_hash_ok = (
            math_review_sheet.exists()
            and math_manual.get("review_sheet_sha256")
            == sha256_file(math_review_sheet)
        )
        math_audit_hash_ok = (
            math_audit_artifact.exists()
            and math_manual.get("automatic_audit_sha256")
            == sha256_file(math_audit_artifact)
        )
        math_manual_pass = (
            math_manual.get("protocol") == "trajuq_math500_manual_scorer_gate_v2"
            and math_manual.get("status") == "PASS"
            and math_manual.get("git_commit") == experiment_commit
            and math_manual.get("automatic_audit_git_commit") == experiment_commit
            and int(math_manual.get("reviewed_records", 0)) >= 25
            and math_review_hash_ok
            and math_audit_hash_ok
            and math_checks.get("answer_extraction") is True
            and math_checks.get("symbolic_equivalence") is True
            and math_checks.get("no_manual_label_flips") is True
        )
        checks["math500_manual_review"] = {
            "path": math500_manual_path,
            "status": math_manual.get("status"),
            "protocol": math_manual.get("protocol"),
            "git_commit": math_manual.get("git_commit"),
            "reviewed_records": math_manual.get("reviewed_records"),
            "review_sheet_hash_ok": math_review_hash_ok,
            "automatic_audit_hash_ok": math_audit_hash_ok,
        }
        if not math_manual_pass:
            failures.append("MATH-500 manual scorer review is incomplete or stale")
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as exc:
        failures.append(f"MATH-500 manual scorer review unavailable/invalid: {exc}")

    parity_checks: dict[str, dict] = {}
    for model in primary_models:
        path = parity_paths.get(model)
        if not path:
            failures.append(f"no parity artifact configured for primary model {model}")
            continue
        try:
            payload = load_json(path)
            status = payload.get("status")
            mismatch_count = payload.get("mismatch_count")
            provenance_mismatch_count = payload.get("provenance_mismatch_count")
            shared = payload.get("shared_records")
            run_provenance = payload.get("run_provenance", {})
            expected_revision = ledger_models.get(model, {}).get("resolved_sha")
            passed = (
                status == "PASS"
                and payload.get("protocol") == "trajuq_exact_parity_v3"
                and payload.get("require_token_ids") is True
                and payload.get("require_provenance") is True
                and experiment_commit is not None
                and payload.get("git_commit") == experiment_commit
                and run_provenance.get("adapter") == model
                and str(run_provenance.get("model_revision")) == str(expected_revision)
                and int(shared or 0) == 20
                and int(mismatch_count or 0) == 0
                and int(provenance_mismatch_count or 0) == 0
            )
            parity_checks[model] = {
                "path": path,
                "status": status,
                "protocol": payload.get("protocol"),
                "require_token_ids": payload.get("require_token_ids"),
                "require_provenance": payload.get("require_provenance"),
                "mismatch_count": mismatch_count,
                "provenance_mismatch_count": provenance_mismatch_count,
                "shared_records": shared,
                "git_commit": payload.get("git_commit"),
                "run_provenance": run_provenance,
                "expected_model_revision": expected_revision,
            }
            if not passed:
                failures.append(
                    f"{model} parity gate is not an exact 20-prompt PASS at the frozen revision/commit"
                )
        except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as exc:
            failures.append(f"{model} parity artifact unavailable/invalid: {exc}")

    checks["model_parity"] = parity_checks
    return {
        "protocol": "trajuq_primary_readiness_v2",
        "status": "PASS" if not failures else "FAIL",
        "primary_config": primary_config,
        "primary_models": primary_models,
        "primary_datasets": primary_datasets,
        "experiment_git_commit": experiment_commit,
        "checks": checks,
        "failures": failures,
        "next_step": (
            "Primary matrix may be launched under frozen configs."
            if not failures
            else "Do not launch the full primary matrix."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hard gate before launching the full TrajUQ primary experiment matrix."
    )
    parser.add_argument(
        "--primary-config",
        default="configs/experiments/primary.yaml",
    )
    parser.add_argument(
        "--cpu-verification",
        default="results/cpu_verification.json",
    )
    parser.add_argument(
        "--revision-ledger",
        default="results/revision_ledger.json",
    )
    parser.add_argument(
        "--pilot-quality",
        default="results/llada_gsm8k_pilot50_gate.json",
    )
    parser.add_argument(
        "--manual-gate",
        default="results/llada_gsm8k_pilot50_manual_review.json",
    )
    parser.add_argument(
        "--math500-scorer-audit",
        default="results/math500_scorer_audit.json",
    )
    parser.add_argument(
        "--math500-manual-review",
        default="results/math500_scorer_manual_review.json",
    )
    parser.add_argument(
        "--parity",
        action="append",
        type=parse_override,
        default=[],
        help="Override MODEL=PATH; may be repeated.",
    )
    parser.add_argument(
        "--output",
        default="results/primary_readiness.json",
    )
    args = parser.parse_args()

    parity_paths = dict(DEFAULT_PARITY)
    parity_paths.update(dict(args.parity))
    payload = evaluate_readiness(
        args.primary_config,
        args.pilot_quality,
        args.manual_gate,
        parity_paths,
        args.math500_scorer_audit,
        args.math500_manual_review,
        args.cpu_verification,
        args.revision_ledger,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
