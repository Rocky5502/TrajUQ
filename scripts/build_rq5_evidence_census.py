#!/usr/bin/env python
"""Create a read-only, outcome-blind RQ5 evidence census and recovery record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def all_nul(path: Path) -> bool:
    data = path.read_bytes()
    return bool(data) and not data.strip(b"\x00")


def tree_binding(root: Path, patterns: tuple[str, ...]) -> dict:
    files = sorted(
        {path for pattern in patterns for path in root.rglob(pattern) if path.is_file()},
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        digest.update(f"{relative}\0{sha256(path)}\0{size}\n".encode("utf-8"))
    return {
        "algorithm": "sha256(sorted(relative_path\\0file_sha256\\0size\\n))",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def repo_binding(root: Path, relative: str, *, rows: bool = False) -> dict:
    path = root / relative
    binding = {"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if rows:
        binding["rows"] = line_count(path)
    return binding


def build(args: argparse.Namespace) -> tuple[dict, dict]:
    root = args.root.resolve()
    freeze_path = root / "results/rq5/TMLR_RQ5_BLACKBOX_FREEZE.json"
    freeze_hash = sha256(freeze_path)
    if freeze_hash != Path(str(freeze_path) + ".sha256").read_text(encoding="utf-8").strip():
        raise SystemExit("RQ5 freeze hash drift")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    registry_path = root / "results/rq5/validated/response_registry.jsonl"
    original_manifest_path = root / "results/rq5/validated/registry_manifest.json"
    registry_hash = sha256(registry_path)
    registry_rows = line_count(registry_path)
    if registry_rows != 58133 or registry_hash != "874176f4b96fca85f1fb843728fdaf94d256d7f62c0ce39dd9e48f8a4a8220ba":
        raise SystemExit("RQ5 canonical registry binding drift")

    reconciliation = json.loads(args.reconciliation_json.read_text(encoding="utf-8"))
    ledger_hash = sha256(args.reconciliation_ledger)
    if ledger_hash != reconciliation["ledger_sha256"] or line_count(args.reconciliation_ledger) != 58260:
        raise SystemExit("RQ5 reconciliation ledger drift")
    expected_counts = {
        "valid": 46681,
        "malformed": 11452,
        "provider_failure": 1,
        "ambiguous_locked": 126,
        "other_terminal": 0,
        "missing_unreconciled": 0,
    }
    if reconciliation["counts"] != expected_counts:
        raise SystemExit("RQ5 terminal accounting drift")
    if reconciliation.get("analysis_authorized") is not False:
        raise SystemExit("terminal reconciliation must not authorize RQ5 claims")

    ambiguous_path = root / "configs/operations/rq5_ambiguous_uuid_registry.yaml"
    ambiguous = yaml.safe_load(ambiguous_path.read_text(encoding="utf-8"))
    explicit_ambiguous_count = len(ambiguous.get("requests", []))
    if explicit_ambiguous_count != reconciliation["explicit_ambiguous_registry_count"]:
        raise SystemExit("RQ5 explicit ambiguous registry drift")

    api_root = root / "results/rq5/api"
    raw_binding = tree_binding(api_root / "records", ("*.json",))
    pending_lock_binding = tree_binding(api_root / "records", ("*.pending",))
    error_binding = tree_binding(api_root / "errors", ("*.json",))
    if raw_binding["file_count"] != 58133:
        raise SystemExit("RQ5 saved response-file count drift")

    census = {
        "schema": "trajuq_rq5_evidence_census_v1",
        "status": "PASS_WITH_PRESERVED_MANIFEST_CORRUPTION",
        "chronology": "POST_DATA_PRE_OUTCOME_ANALYSIS",
        "aggregate_outcomes_computed": False,
        "scientific_claim_authorized": False,
        "network_calls_performed": False,
        "terminal_accounting": {
            "requested_accounted": 58260,
            "saved": 58133,
            "schema_valid": 46681,
            "malformed": 11452,
            "provider_failure": 1,
            "ambiguous_locked_no_replay": 126,
        },
        "bindings": {
            "production_freeze": repo_binding(root, "results/rq5/TMLR_RQ5_BLACKBOX_FREEZE.json"),
            "response_registry": repo_binding(root, "results/rq5/validated/response_registry.jsonl", rows=True),
            "original_registry_manifest": {
                **repo_binding(root, "results/rq5/validated/registry_manifest.json"),
                "all_bytes_nul": all_nul(original_manifest_path),
                "preserved_unchanged": True,
            },
            "prompt_render_ledger": repo_binding(root, "results/rq5/prompt_render_ledger.jsonl", rows=True),
            "provider_cost_ledger": repo_binding(root, "results/rq5/api/cost_ledger.json"),
            "ambiguous_uuid_registry": {
                **repo_binding(root, "configs/operations/rq5_ambiguous_uuid_registry.yaml"),
                "explicitly_registered": explicit_ambiguous_count,
                "terminal_locked_no_replay": reconciliation["counts"]["ambiguous_locked"],
            },
            "raw_response_tree": raw_binding,
            "pending_lock_tree": pending_lock_binding,
            "error_record_tree": error_binding,
            "terminal_reconciliation": {
                "artifact": args.reconciliation_json.name,
                "sha256": sha256(args.reconciliation_json),
            },
            "terminal_ledger": {
                "artifact": args.reconciliation_ledger.name,
                "rows": line_count(args.reconciliation_ledger),
                "sha256": ledger_hash,
            },
            "independent_backup": {
                "artifact": args.backup.name,
                "size_bytes": args.backup.stat().st_size,
                "sha256": sha256(args.backup),
            },
            "provider_snapshot": repo_binding(root, "results/rq5/provider_snapshot/provider_snapshot.json"),
            "pricing_contract": repo_binding(root, "results/rq5/provider_snapshot/pricing_contract.json"),
            "protocol_config": repo_binding(root, "configs/experiments/tmlr_rq5_blackbox.yaml"),
        },
        "frozen_partition_bindings": freeze.get("partitions", []),
        "source_revision": freeze.get("source_commit") or freeze.get("original_authoritative_commit"),
        "firewall": [
            "The original NUL-corrupted registry_manifest.json is preserved byte-for-byte.",
            "This census and its recovered manifest authorize no scientific claim.",
            "No malformed, provider-failure, or ambiguous request may be replayed.",
            "No aggregate correctness or uncertainty outcome was computed during this census.",
        ],
    }
    if census["bindings"]["independent_backup"]["sha256"] != "cc500062ab985e3bb3d7836698c1d1308c5800c76de400de34a97439752d0b6f":
        raise SystemExit("RQ5 independent backup hash drift")

    recovered = {
        "schema": "trajuq_rq5_registry_manifest_recovery_v1",
        "status": "PASS_WITH_PRESERVED_SOURCE_CORRUPTION",
        "partial": False,
        "claim_authorized": False,
        "analysis_authorized": False,
        "recovery_reason": "canonical registry_manifest.json contains only NUL bytes",
        "original_manifest": census["bindings"]["original_registry_manifest"],
        "freeze_sha256": freeze_hash,
        "registry_sha256": registry_hash,
        "validated_responses": 58133,
        "schema_valid_responses": 46681,
        "malformed_responses": 11452,
        "planned_requests": 58260,
        "ambiguous_locked_no_replay": 126,
        "provider_failure": 1,
        "terminal_ledger_sha256": ledger_hash,
        "chronology": "POST_DATA_PRE_OUTCOME_ANALYSIS",
        "note": "Separate recovery artifact; it does not overwrite or reinterpret the original manifest.",
    }
    return census, recovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--reconciliation-json", type=Path, required=True)
    parser.add_argument("--reconciliation-ledger", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/rq5/census/rq5_evidence_census.json"))
    parser.add_argument(
        "--recovered-manifest",
        type=Path,
        default=Path("results/rq5/validated/registry_manifest.recovered.json"),
    )
    args = parser.parse_args()
    census, recovered = build(args)
    root = args.root.resolve()
    for destination, document in ((args.output, census), (args.recovered_manifest, recovered)):
        path = destination if destination.is_absolute() else root / destination
        payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
        write_new(path, payload)
        write_new(Path(str(path) + ".sha256"), (hashlib.sha256(payload).hexdigest() + "\n").encode())


if __name__ == "__main__":
    main()
