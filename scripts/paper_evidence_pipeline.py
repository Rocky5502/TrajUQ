#!/usr/bin/env python
"""Keep operational progress separate from final manuscript evidence.

The ``progress`` command accepts only an operational RQ2 snapshot and emits an
isolated JSON/TeX pair.  The ``final`` command is a fail-closed promotion
trigger: it runs no renderer until all configured RQ manifests pass schema,
provenance, and hash checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/analysis/paper_evidence_pipeline.json"
PROGRESS_SCHEMA = ROOT / "configs/schemas/progress_status_v1.json"
FINAL_SCHEMA = ROOT / "configs/schemas/final_rq_evidence_v1.json"
DEFAULT_PROGRESS_JSON = ROOT / "results/generated/progress_status.json"
DEFAULT_PROGRESS_TEX = ROOT / "paper/generated/progress_status.tex"
DEFAULT_FINAL_REPORT = ROOT / "results/generated/final_readiness.json"

PROGRESS_INPUT_KEYS = {
    "schema",
    "observed_at_utc",
    "state",
    "completed_jobs",
    "required_jobs",
    "completed_seeds",
    "audit_pass_count",
    "throughput_jobs_per_hour",
    "eta_hours",
}
PROGRESS_STATES = {
    "PENDING",
    "RUNNING",
    "BLOCKED",
    "COMPLETE_AWAITING_FINAL_GATES",
}
FORBIDDEN_PROGRESS_TOKENS = {
    "accuracy",
    "auprc",
    "auroc",
    "aurc",
    "brier",
    "calibration",
    "claim_supported",
    "confidence_interval",
    "correctness",
    "ece",
    "effect_size",
    "metric",
    "nll",
    "p_value",
    "risk",
}
HEX = set("0123456789abcdef")


class EvidenceError(ValueError):
    """A fail-closed evidence validation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise EvidenceError(f"evidence path escapes repository: {path}") from exc


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    candidate = path if path.is_absolute() else ROOT / path
    relative(candidate)
    return candidate


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceError(f"missing evidence: {relative(path)}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"invalid JSON evidence {relative(path)}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"evidence root must be an object: {relative(path)}")
    return value


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def progress_input(source: Path) -> dict[str, Any]:
    payload = load_json(source)
    extras = sorted(set(payload) - PROGRESS_INPUT_KEYS)
    if extras:
        raise EvidenceError("progress input contains non-operational fields: " + ", ".join(extras))
    forbidden = sorted(
        key for key in payload for token in FORBIDDEN_PROGRESS_TOKENS if token in key.lower()
    )
    if forbidden:
        raise EvidenceError("progress input contains scientific-result fields: " + ", ".join(forbidden))
    if payload.get("schema") != "trajuq_rq2_progress_input_v1":
        raise EvidenceError("progress input schema must be trajuq_rq2_progress_input_v1")
    if payload.get("state") not in PROGRESS_STATES:
        raise EvidenceError("progress input state is invalid")
    if not isinstance(payload.get("observed_at_utc"), str) or not payload["observed_at_utc"]:
        raise EvidenceError("progress input requires observed_at_utc")
    completed = payload.get("completed_jobs")
    required = payload.get("required_jobs")
    if not isinstance(completed, int) or not isinstance(required, int):
        raise EvidenceError("completed_jobs and required_jobs must be integers")
    if required != 120:
        raise EvidenceError("frozen RQ2 progress denominator must be 120 required seed jobs")
    if completed < 0 or completed > required:
        raise EvidenceError("completed_jobs must lie within [0, required_jobs]")
    seeds = payload.get("completed_seeds", [])
    if not isinstance(seeds, list) or any(not isinstance(seed, int) for seed in seeds):
        raise EvidenceError("completed_seeds must be an integer list")
    if len(seeds) != len(set(seeds)):
        raise EvidenceError("completed_seeds contains duplicates")
    audits = payload.get("audit_pass_count", 0)
    if not isinstance(audits, int) or audits < 0 or audits > completed:
        raise EvidenceError("audit_pass_count must lie within [0, completed_jobs]")
    for key in ("throughput_jobs_per_hour", "eta_hours"):
        value = payload.get(key)
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            raise EvidenceError(f"{key} must be null or non-negative")
    return payload


def tex_escape(value: Any) -> str:
    text = str(value)
    for old, new in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%")):
        text = text.replace(old, new)
    return text


def render_progress_tex(payload: dict[str, Any], source: Path) -> str:
    completed = int(payload["completed_jobs"])
    required = int(payload["required_jobs"])
    percent = 100.0 * completed / required
    seeds = ",".join(str(seed) for seed in payload.get("completed_seeds", [])) or "none"
    throughput = payload.get("throughput_jobs_per_hour")
    eta = payload.get("eta_hours")
    return "\n".join(
        [
            "% AUTO-GENERATED OPERATIONAL PROGRESS; NOT SCIENTIFIC EVIDENCE.",
            "% This file must never be input by a submission manuscript.",
            f"% source={relative(source)} sha256={sha256_file(source)}",
            "\\providecommand{\\RQTwoProgressScienceAuthorization}{NOT AUTHORIZED FOR SCIENTIFIC CLAIMS}",
            f"\\providecommand{{\\RQTwoProgressState}}{{{tex_escape(payload['state'])}}}",
            f"\\providecommand{{\\RQTwoProgressCompletedJobs}}{{{completed}}}",
            f"\\providecommand{{\\RQTwoProgressRequiredJobs}}{{{required}}}",
            f"\\providecommand{{\\RQTwoProgressRemainingJobs}}{{{required - completed}}}",
            f"\\providecommand{{\\RQTwoProgressPercent}}{{{percent:.2f}}}",
            f"\\providecommand{{\\RQTwoProgressCompletedSeeds}}{{{tex_escape(seeds)}}}",
            f"\\providecommand{{\\RQTwoProgressAuditPassCount}}{{{int(payload.get('audit_pass_count', 0))}}}",
            f"\\providecommand{{\\RQTwoProgressThroughput}}{{{'unknown' if throughput is None else f'{float(throughput):.2f}'}}}",
            f"\\providecommand{{\\RQTwoProgressETAHours}}{{{'unknown' if eta is None else f'{float(eta):.2f}'}}}",
            f"\\providecommand{{\\RQTwoProgressObservedAt}}{{{tex_escape(payload['observed_at_utc'])}}}",
            "",
        ]
    )


def build_progress(source: Path, output_json: Path, output_tex: Path, registry_version: str) -> dict[str, Any]:
    payload = progress_input(source)
    tex = render_progress_tex(payload, source).encode("utf-8")
    completed = int(payload["completed_jobs"])
    required = int(payload["required_jobs"])
    result = {
        "schema": "trajuq_progress_status_v1",
        "registry_version": registry_version,
        "scope": "RQ2_OPERATIONAL_PROGRESS_ONLY",
        "state": payload["state"],
        "observed_at_utc": payload["observed_at_utc"],
        "completion": {
            "completed_jobs": completed,
            "required_jobs": required,
            "remaining_jobs": required - completed,
            "completion_percent": round(100.0 * completed / required, 6),
            "completed_seeds": sorted(payload.get("completed_seeds", [])),
            "audit_pass_count": int(payload.get("audit_pass_count", 0)),
            "throughput_jobs_per_hour": payload.get("throughput_jobs_per_hour"),
            "eta_hours": payload.get("eta_hours"),
        },
        "source": {"path": relative(source), "sha256": sha256_file(source)},
        "schema_binding": {"path": relative(PROGRESS_SCHEMA), "sha256": sha256_file(PROGRESS_SCHEMA)},
        "output": {
            "tex_path": relative(output_tex),
            "tex_sha256": hashlib.sha256(tex).hexdigest(),
        },
        "science_authorization": "NOT_AUTHORIZED_FOR_SCIENTIFIC_CLAIMS",
    }
    atomic_write(output_tex, tex)
    atomic_write(output_json, json_bytes(result))
    return result


def validate_artifact_list(value: Any, *, label: str) -> list[str]:
    failures: list[str] = []
    if not isinstance(value, list) or not value:
        return [f"{label} must be a non-empty artifact list"]
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            failures.append(f"{label}[{index}] must contain only path and sha256")
            continue
        try:
            path = repo_path(str(item["path"]))
        except EvidenceError as exc:
            failures.append(str(exc))
            continue
        if not path.is_file():
            failures.append(f"{label}[{index}] missing: {relative(path)}")
        elif not is_sha256(item["sha256"]):
            failures.append(f"{label}[{index}] has invalid SHA-256")
        elif sha256_file(path) != item["sha256"]:
            failures.append(f"{label}[{index}] hash mismatch: {relative(path)}")
    return failures


def validate_final_manifest(path: Path, rq: str, registry_version: str) -> list[str]:
    try:
        payload = load_json(path)
    except EvidenceError as exc:
        return [str(exc)]
    required = {
        "schema", "registry_version", "rq", "status", "experiment_commit",
        "generated_at_utc", "analysis_script", "analysis_script_sha256",
        "audit_status", "identifiers", "gates", "sources", "outputs",
    }
    failures: list[str] = []
    if set(payload) != required:
        failures.append("manifest fields differ from frozen final schema")
    checks = {
        "schema": "trajuq_final_rq_evidence_v1",
        "registry_version": registry_version,
        "rq": rq,
        "status": "FINAL",
        "audit_status": "PASS",
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            failures.append(f"{key} must equal {expected!r}")
    commit = payload.get("experiment_commit")
    if not isinstance(commit, str) or len(commit) != 40 or not set(commit) <= HEX:
        failures.append("experiment_commit must be a 40-character lowercase hex commit")
    if not isinstance(payload.get("generated_at_utc"), str) or not payload["generated_at_utc"]:
        failures.append("generated_at_utc is required")
    try:
        script = repo_path(str(payload.get("analysis_script", "")))
        if not script.is_file():
            failures.append(f"analysis script missing: {relative(script)}")
        elif not is_sha256(payload.get("analysis_script_sha256")):
            failures.append("analysis_script_sha256 is invalid")
        elif sha256_file(script) != payload["analysis_script_sha256"]:
            failures.append("analysis script hash mismatch")
    except EvidenceError as exc:
        failures.append(str(exc))
    identifiers = payload.get("identifiers")
    if not isinstance(identifiers, dict) or set(identifiers) != {"models", "datasets", "seeds"}:
        failures.append("identifiers must contain exactly models, datasets, and seeds")
    else:
        for key in ("models", "datasets", "seeds"):
            values = identifiers[key]
            if not isinstance(values, list) or not values or len(values) != len(set(values)):
                failures.append(f"identifiers.{key} must be a non-empty unique list")
    gates = payload.get("gates")
    required_gates = {
        "generation_complete",
        "audits_pass",
        "uniqueness_pass",
        "completeness_pass",
        "scoring_complete",
        "analysis_complete",
        "claim_gate_decided",
        "registry_frozen",
    }
    if not isinstance(gates, dict) or not required_gates <= set(gates):
        failures.append("gates must contain every frozen finalization gate")
    else:
        for gate in sorted(required_gates):
            if gates.get(gate) is not True:
                failures.append(f"gates.{gate} must be true")
        if rq == "RQ2":
            if gates.get("seed_jobs_expected") != 120:
                failures.append("RQ2 gates.seed_jobs_expected must equal 120")
            if gates.get("seed_jobs_complete") != 120:
                failures.append("RQ2 gates.seed_jobs_complete must equal 120")
    failures.extend(validate_artifact_list(payload.get("sources"), label="sources"))
    failures.extend(validate_artifact_list(payload.get("outputs"), label="outputs"))
    return failures


def final_readiness(config_path: Path, report_path: Path, *, check_only: bool) -> tuple[dict[str, Any], int]:
    config = load_json(config_path)
    if config.get("schema") != "trajuq_paper_evidence_pipeline_config_v1":
        raise EvidenceError("paper evidence pipeline config schema is not recognized")
    registry_version = str(config.get("registry_version", ""))
    required_rqs = config.get("required_rqs")
    manifests = config.get("final_manifests")
    if required_rqs != ["RQ1", "RQ2", "RQ3", "RQ4", "RQ5"] or not isinstance(manifests, dict):
        raise EvidenceError("pipeline must require the frozen RQ1--RQ5 set")
    checks: dict[str, Any] = {}
    for rq in required_rqs:
        manifest = repo_path(str(manifests.get(rq, "")))
        failures = validate_final_manifest(manifest, rq, registry_version)
        checks[rq] = {
            "manifest": relative(manifest),
            "manifest_sha256": sha256_file(manifest) if manifest.is_file() else None,
            "status": "PASS" if not failures else "LOCKED",
            "failures": failures,
        }
    blockers = [rq for rq, check in checks.items() if check["status"] != "PASS"]
    report = {
        "schema": "trajuq_final_paper_readiness_v1",
        "registry_version": registry_version,
        "config": {"path": relative(config_path), "sha256": sha256_file(config_path)},
        "final_schema": {"path": relative(FINAL_SCHEMA), "sha256": sha256_file(FINAL_SCHEMA)},
        "state": "LOCKED" if blockers else "READY",
        "promotion_executed": False,
        "checks": checks,
        "blockers": blockers,
    }
    atomic_write(report_path, json_bytes(report))
    if blockers:
        return report, 2
    if check_only:
        return report, 0
    commands = config.get("promotion_commands")
    if not isinstance(commands, list) or not commands:
        raise EvidenceError("promotion_commands must be a non-empty list")
    executed: list[list[str]] = []
    for command in commands:
        if not isinstance(command, list) or not command or any(not isinstance(v, str) for v in command):
            raise EvidenceError("each promotion command must be a non-empty string list")
        resolved = [sys.executable, *command[1:]] if command[0] == "python" else command
        subprocess.run(resolved, cwd=ROOT, check=True)
        executed.append(command)
    report["state"] = "PROMOTED"
    report["promotion_executed"] = True
    report["executed_commands"] = executed
    atomic_write(report_path, json_bytes(report))
    return report, 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    progress = commands.add_parser("progress", help="render isolated operational progress")
    progress.add_argument("--source", required=True)
    progress.add_argument("--output-json", default=str(DEFAULT_PROGRESS_JSON))
    progress.add_argument("--output-tex", default=str(DEFAULT_PROGRESS_TEX))
    progress.add_argument("--config", default=str(DEFAULT_CONFIG))
    final = commands.add_parser("final", help="validate all RQs, then promote atomically gated science")
    final.add_argument("--config", default=str(DEFAULT_CONFIG))
    final.add_argument("--report", default=str(DEFAULT_FINAL_REPORT))
    final.add_argument("--check-only", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "progress":
            config = load_json(repo_path(args.config))
            result = build_progress(
                repo_path(args.source),
                repo_path(args.output_json),
                repo_path(args.output_tex),
                str(config["registry_version"]),
            )
            print(json.dumps({"state": result["state"], "science_authorization": result["science_authorization"]}))
            return 0
        result, code = final_readiness(
            repo_path(args.config), repo_path(args.report), check_only=args.check_only
        )
        print(json.dumps({"state": result["state"], "blockers": result["blockers"]}))
        return code
    except (EvidenceError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"PAPER EVIDENCE PIPELINE ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
