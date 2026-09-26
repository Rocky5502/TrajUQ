#!/usr/bin/env python
from __future__ import annotations

import argparse
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
) -> dict:
    failures: list[str] = []
    checks: dict[str, dict] = {}

    cfg_path = Path(primary_config)
    if not cfg_path.exists():
        failures.append(f"missing primary config: {primary_config}")
        primary_models: list[str] = []
    else:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        primary_models = [
            Path(path).stem
            for path in cfg.get("models", {}).get("primary", [])
        ]
        if not primary_models:
            failures.append("primary config contains no primary models")

    try:
        cpu = load_json(cpu_verification_path)
        cpu_steps = cpu.get("steps", [])
        cpu_pass = (
            cpu.get("status") == "PASS"
            and cpu.get("git_clean") is True
            and len(cpu_steps) == len(cpu.get("expected_steps", []))
            and all(step.get("status") == "PASS" for step in cpu_steps)
        )
        checks["cpu_verification"] = {
            "path": cpu_verification_path,
            "status": cpu.get("status"),
            "git_commit": cpu.get("git_commit"),
            "git_clean": cpu.get("git_clean"),
            "completed_steps": cpu.get("completed_steps"),
        }
        if not cpu_pass:
            failures.append("CPU repository verification is not a clean PASS")
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as exc:
        failures.append(f"CPU verification artifact unavailable/invalid: {exc}")

    try:
        quality = load_json(pilot_quality_path)
        quality_pass = quality.get("status") == "PASS"
        checks["pilot_quality"] = {
            "path": pilot_quality_path,
            "status": quality.get("status"),
            "records": quality.get("raw_records"),
        }
        if not quality_pass:
            failures.append("50-example pilot quality gate is not PASS")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        failures.append(f"pilot quality artifact unavailable/invalid: {exc}")

    try:
        manual = load_json(manual_gate_path)
        manual_checks = manual.get("checks", {})
        required_manual = (
            manual.get("status") == "PASS"
            and int(manual.get("reviewed_records", 0)) >= 10
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
            "reviewed_records": manual.get("reviewed_records"),
        }
        if not required_manual:
            failures.append("human pilot-review gate is incomplete")
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as exc:
        failures.append(f"manual pilot artifact unavailable/invalid: {exc}")

    try:
        math_audit = load_json(math500_audit_path)
        math_audit_pass = (
            math_audit.get("status") == "PASS"
            and int(math_audit.get("failure_count", 1)) == 0
        )
        checks["math500_scorer"] = {
            "path": math500_audit_path,
            "status": math_audit.get("status"),
            "failure_count": math_audit.get("failure_count"),
        }
        if not math_audit_pass:
            failures.append("MATH-500 automatic scorer audit is not PASS")
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as exc:
        failures.append(f"MATH-500 scorer audit unavailable/invalid: {exc}")

    try:
        math_manual = load_json(math500_manual_path)
        math_checks = math_manual.get("checks", {})
        math_manual_pass = (
            math_manual.get("status") == "PASS"
            and int(math_manual.get("reviewed_records", 0)) >= 25
            and math_checks.get("answer_extraction") is True
            and math_checks.get("symbolic_equivalence") is True
            and math_checks.get("no_manual_label_flips") is True
        )
        checks["math500_manual_review"] = {
            "path": math500_manual_path,
            "status": math_manual.get("status"),
            "reviewed_records": math_manual.get("reviewed_records"),
        }
        if not math_manual_pass:
            failures.append("MATH-500 manual scorer review is incomplete")
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
            passed = (
                status == "PASS"
                and payload.get("protocol") == "trajuq_exact_parity_v2"
                and payload.get("require_token_ids") is True
                and payload.get("require_provenance") is True
            )
            if mismatch_count is not None:
                passed = passed and int(mismatch_count) == 0
            if provenance_mismatch_count is not None:
                passed = passed and int(provenance_mismatch_count) == 0
            else:
                passed = False
            if shared is not None:
                passed = passed and int(shared) > 0
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
            }
            if not passed:
                failures.append(f"{model} parity gate is not PASS")
        except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError) as exc:
            failures.append(f"{model} parity artifact unavailable/invalid: {exc}")

    checks["model_parity"] = parity_checks
    return {
        "protocol": "trajuq_primary_readiness_v1",
        "status": "PASS" if not failures else "FAIL",
        "primary_config": primary_config,
        "primary_models": primary_models,
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
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
