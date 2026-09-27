#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from trajuq.datasets import NORMALIZER_VERSION, SPLIT_SELECTION_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen_preprimary(
    *,
    cpu_path: Path = Path("results/cpu_verification.json"),
    ledger_path: Path = Path("results/revision_ledger.json"),
    gated_path: Path = Path("results/gated_dataset_access.json"),
    contracts_path: Path = Path("results/dataset_contracts.json"),
    require_current_checkout: bool = True,
    current_head: str | None = None,
    current_clean: bool | None = None,
) -> dict[str, Any]:
    failures: list[str] = []

    paths = {
        "cpu_verification": cpu_path,
        "revision_ledger": ledger_path,
        "gated_dataset_access": gated_path,
        "dataset_contracts": contracts_path,
    }
    for name, path in paths.items():
        if not path.is_file():
            failures.append(f"missing {name} artifact: {path}")

    if failures:
        return {
            "protocol": "trajuq_frozen_preprimary_validation_v1",
            "status": "FAIL",
            "git_commit": current_head,
            "failures": failures,
            "checks": {},
        }

    try:
        cpu = load_json(cpu_path)
        ledger = load_json(ledger_path)
        gated = load_json(gated_path)
        contracts = load_json(contracts_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "protocol": "trajuq_frozen_preprimary_validation_v1",
            "status": "FAIL",
            "git_commit": current_head,
            "failures": [f"invalid pre-primary JSON artifact: {type(exc).__name__}"],
            "checks": {},
        }

    experiment_commit = str(ledger.get("git_commit", "")).strip() or None
    if require_current_checkout:
        if current_head is None:
            current_head = git_text("rev-parse", "HEAD")
        if current_clean is None:
            status = git_text("status", "--porcelain")
            current_clean = status == "" if status is not None else None
        if not current_head:
            failures.append("current git commit could not be resolved")
        elif current_head != experiment_commit:
            failures.append(
                "current git commit differs from frozen experiment commit"
            )
        if current_clean is not True:
            failures.append("current git worktree is not clean")

    primary_path = Path(str(ledger.get("primary_config", "")))
    manifest_path = Path(str(ledger.get("dataset_manifest", "")))
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

    ledger_ok = (
        ledger.get("protocol") == "trajuq_revision_ledger_v2"
        and ledger.get("status") == "PASS"
        and bool(experiment_commit)
        and primary_path.is_file()
        and manifest_path.is_file()
        and ledger.get("primary_config_sha256") == (
            sha256_file(primary_path) if primary_path.is_file() else None
        )
        and ledger.get("dataset_manifest_sha256") == (
            sha256_file(manifest_path) if manifest_path.is_file() else None
        )
        and bool(ledger_models)
        and bool(ledger_datasets)
    )
    if not ledger_ok:
        failures.append("revision ledger/config provenance is not a valid PASS")

    manifest_definitions: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_definitions = dict(manifest_payload.get("datasets", {}))
        except (OSError, json.JSONDecodeError, TypeError):
            failures.append("dataset manifest is invalid")

    primary_models: set[str] = set()
    primary_datasets: set[str] = set()
    if primary_path.is_file():
        try:
            primary = yaml.safe_load(primary_path.read_text(encoding="utf-8"))
            primary_models = {
                Path(str(path)).stem
                for path in primary.get("models", {}).get("primary", [])
            }
            primary_datasets = {
                str(name) for name in primary.get("datasets", [])
            }
        except (OSError, yaml.YAMLError, TypeError):
            failures.append("primary config is invalid")
    if primary_models and set(ledger_models) != primary_models:
        failures.append("revision ledger model set differs from primary config")
    if primary_datasets and set(ledger_datasets) != primary_datasets:
        failures.append("revision ledger dataset set differs from primary config")
    if primary_datasets and not primary_datasets.issubset(set(manifest_definitions)):
        failures.append("primary dataset set is missing from dataset manifest")
    for name, row in ledger_datasets.items():
        spec = manifest_definitions.get(name)
        if not isinstance(spec, dict):
            failures.append(f"{name}: dataset manifest definition is missing")
            continue
        if str(row.get("repo_id", "")) != str(spec.get("hf_id", "")):
            failures.append(
                f"{name}: revision-ledger repo_id differs from dataset manifest"
            )
        cap_spec = spec.get("pre_generation_cap")
        if not isinstance(cap_spec, dict):
            failures.append(f"{name}: dataset cap policy is missing")
            continue
        if cap_spec.get("method") != SPLIT_SELECTION_VERSION:
            failures.append(f"{name}: dataset cap method is not frozen")
        if not str(cap_spec.get("seed", "")).strip():
            failures.append(f"{name}: dataset cap seed is missing")
        for key in ("calibration_max", "test_max"):
            value = cap_spec.get(key)
            if value is None:
                continue
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                failures.append(f"{name}: {key} is invalid")
                continue
            if parsed <= 0:
                failures.append(f"{name}: {key} must be positive")

    frozen_paths = ledger.get("frozen_model_configs", {})
    frozen_hashes = ledger.get("frozen_model_config_sha256", {})
    frozen_model_checks: dict[str, bool] = {}
    for adapter, row in ledger_models.items():
        path = Path(str(frozen_paths.get(adapter, "")))
        expected_hash = str(frozen_hashes.get(adapter, ""))
        ok = path.is_file() and bool(expected_hash)
        if ok:
            ok = sha256_file(path) == expected_hash
        if ok:
            try:
                frozen = yaml.safe_load(path.read_text(encoding="utf-8"))
                ok = (
                    str(frozen.get("adapter", "")) == adapter
                    and str(frozen.get("model_id", ""))
                    == str(row.get("model_id", ""))
                    and str(frozen.get("revision", ""))
                    == str(row.get("resolved_sha", ""))
                    and str(frozen.get("device", "")) == "cuda"
                    and str(frozen.get("dtype", "")) == "bfloat16"
                )
            except (OSError, yaml.YAMLError, TypeError):
                ok = False
        frozen_model_checks[adapter] = bool(ok)
        if not ok:
            failures.append(
                f"{adapter}: frozen model config does not match revision ledger"
            )

    cpu_steps = list(cpu.get("steps", []))
    expected_steps = list(cpu.get("expected_steps", []))
    completed_steps = list(cpu.get("completed_steps", []))
    step_names = [step.get("name") for step in cpu_steps]
    cpu_logs_ok = True
    for step in cpu_steps:
        log = Path(str(step.get("log", "")))
        expected_hash = str(step.get("log_sha256", ""))
        if (
            not log.is_file()
            or not expected_hash
            or sha256_file(log) != expected_hash
        ):
            cpu_logs_ok = False
            break
    cpu_ok = (
        bool(expected_steps)
        and cpu.get("protocol") == "trajuq_cpu_verification_v1"
        and cpu.get("status") == "PASS"
        and cpu.get("git_clean") is True
        and cpu.get("git_commit") == experiment_commit
        and completed_steps == expected_steps
        and step_names == expected_steps
        and len(cpu_steps) == len(expected_steps)
        and all(step.get("status") == "PASS" for step in cpu_steps)
        and cpu_logs_ok
    )
    if not cpu_ok:
        failures.append("CPU verification/log evidence is not a clean frozen PASS")

    ledger_sha = sha256_file(ledger_path)
    expected_gated = sorted(
        name for name, row in ledger_datasets.items() if row.get("gated") is True
    )
    gated_rows = {
        str(item.get("name")): item
        for item in gated.get("datasets", [])
        if item.get("name")
    }
    gated_ok = (
        gated.get("protocol") == "trajuq_gated_dataset_access_v1"
        and gated.get("status") == "PASS"
        and gated.get("git_clean") is True
        and gated.get("git_commit") == experiment_commit
        and gated.get("revision_ledger_sha256") == ledger_sha
        and sorted(str(x) for x in gated.get("gated_dataset_names", []))
        == expected_gated
        and set(gated_rows) == set(expected_gated)
        and int(gated.get("failure_count", 1)) == 0
        and all(
            gated_rows[name].get("status") == "PASS"
            and gated_rows[name].get("record_accessed") is True
            and str(gated_rows[name].get("repo_id", ""))
            == str(ledger_datasets[name].get("repo_id", ""))
            and str(gated_rows[name].get("resolved_sha", ""))
            == str(ledger_datasets[name].get("resolved_sha", ""))
            for name in expected_gated
        )
    )
    if not gated_ok:
        failures.append("gated dataset access is not a revision-bound PASS")

    expected_contracts = sorted(ledger_datasets)
    observed_contracts = sorted(
        str(x) for x in contracts.get("observed_datasets", [])
    )
    declared_contracts = sorted(
        str(x) for x in contracts.get("expected_datasets", [])
    )
    contract_rows = {
        str(item.get("dataset")): item
        for item in contracts.get("results", [])
        if item.get("dataset")
    }
    required_contract_samples = 4
    contracts_ok = (
        contracts.get("protocol") == "trajuq_dataset_contracts_v1"
        and contracts.get("status") == "PASS"
        and contracts.get("git_clean") is True
        and contracts.get("git_commit") == experiment_commit
        and contracts.get("revision_ledger_sha256") == ledger_sha
        and contracts.get("normalizer_version") == NORMALIZER_VERSION
        and contracts.get("selection_version") == SPLIT_SELECTION_VERSION
        and int(
            contracts.get("contract_sample_count_per_dataset", 0)
        ) >= required_contract_samples
        and declared_contracts == expected_contracts
        and observed_contracts == expected_contracts
        and set(contract_rows) == set(expected_contracts)
        and int(contracts.get("failure_count", 1)) == 0
        and all(
            contract_rows[name].get("status") == "PASS"
            and contract_rows[name].get("scorer_smoke_pass") is True
            and int(
                contract_rows[name].get("contract_sample_count", 0)
            ) >= required_contract_samples
            and int(
                contract_rows[name].get("observed_sample_count", 0)
            ) >= required_contract_samples
            and int(
                contract_rows[name].get("scorer_smoke_pass_count", 0)
            ) >= required_contract_samples
            and contract_rows[name].get("source_row_content_stored") is False
            and contract_rows[name].get("normalizer_version")
            == NORMALIZER_VERSION
            and str(contract_rows[name].get("repo_id", ""))
            == str(ledger_datasets[name].get("repo_id", ""))
            and str(contract_rows[name].get("resolved_revision", ""))
            == str(ledger_datasets[name].get("resolved_sha", ""))
            for name in expected_contracts
        )
    )
    if not contracts_ok:
        failures.append(
            "dataset normalizer/scorer contracts are not a complete revision-bound PASS"
        )

    return {
        "protocol": "trajuq_frozen_preprimary_validation_v1",
        "status": "PASS" if not failures else "FAIL",
        "git_commit": experiment_commit,
        "checks": {
            "revision_ledger": ledger_ok,
            "frozen_model_configs": frozen_model_checks,
            "cpu_verification": cpu_ok,
            "cpu_logs": cpu_logs_ok,
            "gated_dataset_access": gated_ok,
            "dataset_contracts": contracts_ok,
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate reusable frozen pre-primary evidence before any GPU "
            "parity/pilot/MATH stage."
        )
    )
    parser.add_argument(
        "--historical-provenance-only",
        action="store_true",
        help=(
            "Do not require the current checkout to equal the historical "
            "experiment commit. Intended for later audit tooling, not GPU runs."
        ),
    )
    args = parser.parse_args()

    payload = validate_frozen_preprimary(
        require_current_checkout=not args.historical_provenance_only,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
