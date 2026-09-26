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
            shared = payload.get("shared_records")
            passed = status == "PASS"
            if mismatch_count is not None:
                passed = passed and int(mismatch_count) == 0
            if shared is not None:
                passed = passed and int(shared) > 0
            parity_checks[model] = {
                "path": path,
                "status": status,
                "mismatch_count": mismatch_count,
                "shared_records": shared,
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
        "--pilot-quality",
        default="results/llada_gsm8k_pilot50_gate.json",
    )
    parser.add_argument(
        "--manual-gate",
        default="results/llada_gsm8k_pilot50_manual_review.json",
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
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
