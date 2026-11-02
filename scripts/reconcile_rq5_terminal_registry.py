#!/usr/bin/env python
"""Fail-closed, metadata-only reconciliation for the frozen RQ5 request universe.

This command never contacts a provider.  In ``--check-only`` mode it performs
no writes.  In materialization mode it writes a new terminal ledger and
manifest, refusing to overwrite either path.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import yaml


NAMESPACE = uuid.UUID("6ddf4f6d-50dc-4a38-94d3-94ace947ff7c")
CATEGORIES = (
    "valid",
    "malformed",
    "provider_failure",
    "ambiguous_locked",
    "other_terminal",
    "missing_unreconciled",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def expected_requests(root: Path, freeze: dict, freeze_sha: str) -> dict[str, dict]:
    requests: dict[str, dict] = {}
    repeats = int(freeze["decoding"]["repeated_samples"])
    for model in freeze["models"]:
        for partition in freeze["partitions"]:
            dataset = str(partition["dataset"])
            split = str(partition["split"])
            data_path = root / "data" / "processed" / dataset / f"{split}.jsonl"
            if not data_path.is_file():
                raise FileNotFoundError(f"frozen partition is missing: {data_path}")
            if sha256_path(data_path) != partition["sha256"]:
                raise ValueError(f"frozen partition hash drift: {data_path}")
            # JSONL is delimited by LF. ``str.splitlines`` also splits on
            # Unicode separators that may legitimately occur inside a prompt.
            with data_path.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            if len(rows) != int(partition["rows"]):
                raise ValueError(f"frozen partition row-count drift: {data_path}")
            for row in rows:
                example_id = str(row["id"])
                for repeat_index in range(repeats):
                    request_uuid = str(
                        uuid.uuid5(
                            NAMESPACE,
                            f"{freeze_sha}|{model}|{dataset}|{split}|{example_id}|{repeat_index}",
                        )
                    )
                    if request_uuid in requests:
                        raise ValueError(f"duplicate expected request UUID: {request_uuid}")
                    requests[request_uuid] = {
                        "request_uuid": request_uuid,
                        "model": str(model),
                        "dataset": dataset,
                        "split": split,
                        "example_id": example_id,
                        "repeat_index": repeat_index,
                    }
    return requests


def files_by_stem(root: Path, pattern: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.glob(pattern):
        if path.stem in result:
            raise ValueError(f"duplicate artifact stem {path.stem}: {result[path.stem]} and {path}")
        result[path.stem] = path
    return result


def record_reasons(
    record: dict, identity: dict, freeze_sha: str, required: set[str], forbidden: set[str]
) -> list[str]:
    reasons: list[str] = []
    if required - set(record):
        reasons.append("missing_required_fields")
    if forbidden & set(record):
        reasons.append("forbidden_fields")
    checks = {
        "freeze_hash_mismatch": record.get("freeze_sha256") != freeze_sha,
        "request_uuid_mismatch": record.get("request_uuid") != identity["request_uuid"],
        "requested_model_mismatch": record.get("requested_model") != identity["model"],
        "returned_model_mismatch": record.get("returned_model") != identity["model"],
        "dataset_mismatch": record.get("dataset") != identity["dataset"],
        "split_mismatch": record.get("split") != identity["split"],
        "example_id_mismatch": str(record.get("example_id")) != identity["example_id"],
        "repeat_index_mismatch": record.get("repeat_index") != identity["repeat_index"],
        "non_200_success_record": record.get("http_status") != 200,
    }
    reasons.extend(name for name, failed in checks.items() if failed)
    answer = record.get("parsed_final_answer")
    confidence = record.get("verbalized_confidence")
    if not isinstance(answer, str) or not answer.strip():
        reasons.append("missing_final_answer")
    if (
        not isinstance(confidence, int)
        or isinstance(confidence, bool)
        or not 0 <= confidence <= 100
    ):
        reasons.append("missing_or_invalid_confidence")
    return sorted(set(reasons))


def error_reasons(error: dict, identity: dict, freeze_sha: str) -> list[str]:
    checks = {
        "error_schema_mismatch": error.get("schema") != "trajuq_rq5_request_error_v1",
        "freeze_hash_mismatch": error.get("freeze_sha256") != freeze_sha,
        "request_uuid_mismatch": error.get("request_uuid") != identity["request_uuid"],
        "requested_model_mismatch": error.get("requested_model") != identity["model"],
        "dataset_mismatch": error.get("dataset") != identity["dataset"],
        "split_mismatch": error.get("split") != identity["split"],
        "example_id_mismatch": str(error.get("example_id")) != identity["example_id"],
        "repeat_index_mismatch": error.get("repeat_index") != identity["repeat_index"],
        "error_marked_for_resubmission": error.get("will_resubmit") is not False,
    }
    return sorted(name for name, failed in checks.items() if failed)


def reconcile(
    *,
    root: Path,
    freeze_path: Path,
    schema_path: Path,
    api_root: Path,
    ambiguous_registry_path: Path,
) -> tuple[dict, list[dict]]:
    freeze = read_json(freeze_path)
    freeze_sha = sha256_path(freeze_path)
    sidecar = Path(str(freeze_path) + ".sha256")
    if (
        freeze.get("status") != "PASS"
        or not sidecar.is_file()
        or sidecar.read_text().strip() != freeze_sha
    ):
        raise ValueError("RQ5 freeze is not PASS or its SHA-256 sidecar drifted")
    schema = read_json(schema_path)
    required, forbidden = set(schema["required"]), set(schema.get("forbidden", []))
    expected = expected_requests(root, freeze, freeze_sha)

    ambiguous_contract = yaml.safe_load(ambiguous_registry_path.read_text(encoding="utf-8"))
    if ambiguous_contract.get("freeze_sha256") != freeze_sha:
        raise ValueError("ambiguous UUID registry is not bound to the active freeze")
    explicit_ambiguous = {
        str(item["request_uuid"]) for item in ambiguous_contract.get("requests", [])
    }

    records = files_by_stem(api_root / "records", "*/*/*/*.json")
    errors = files_by_stem(api_root / "errors", "*/*/*/*.json")
    pending = files_by_stem(api_root / "records", "*/*/*/*.pending")
    expected_ids = set(expected)
    orphans = {
        "record_files": sorted(set(records) - expected_ids),
        "error_files": sorted(set(errors) - expected_ids),
        "pending_files": sorted(set(pending) - expected_ids),
        "explicit_ambiguous": sorted(explicit_ambiguous - expected_ids),
    }

    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    cells: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    ledger: list[dict] = []
    overlaps = 0
    provenance_violations = 0
    for request_uuid, identity in sorted(expected.items()):
        artifacts = [request_uuid in records, request_uuid in errors, request_uuid in pending]
        if sum(artifacts) > 1 and not (
            request_uuid in errors and request_uuid in pending and request_uuid not in records
        ):
            overlaps += 1
        reasons: list[str] = []
        artifact_paths: list[str] = []
        artifact_hashes: list[str] = []
        if request_uuid in records:
            path = records[request_uuid]
            artifact_paths.append(path.relative_to(root).as_posix())
            artifact_hashes.append(sha256_path(path))
            try:
                reasons = record_reasons(read_json(path), identity, freeze_sha, required, forbidden)
            except Exception:
                reasons = ["unreadable_record_json"]
            category = "malformed" if reasons else "valid"
            if any(
                reason not in {"missing_final_answer", "missing_or_invalid_confidence"}
                for reason in reasons
            ):
                provenance_violations += 1
        elif request_uuid in errors:
            path = errors[request_uuid]
            artifact_paths.append(path.relative_to(root).as_posix())
            artifact_hashes.append(sha256_path(path))
            try:
                error = read_json(path)
                reasons = error_reasons(error, identity, freeze_sha)
                if error.get("confirmed_provider_error") is True:
                    category = "provider_failure"
                elif request_uuid in pending or request_uuid in explicit_ambiguous:
                    category = "ambiguous_locked"
                else:
                    category = "other_terminal"
            except Exception:
                category, reasons = "other_terminal", ["unreadable_error_json"]
            if reasons:
                provenance_violations += 1
            if request_uuid in pending:
                lock = pending[request_uuid]
                artifact_paths.append(lock.relative_to(root).as_posix())
                artifact_hashes.append(sha256_path(lock))
        elif request_uuid in pending or request_uuid in explicit_ambiguous:
            category = "ambiguous_locked"
            if request_uuid in pending:
                lock = pending[request_uuid]
                artifact_paths.append(lock.relative_to(root).as_posix())
                artifact_hashes.append(sha256_path(lock))
        else:
            category = "missing_unreconciled"

        counts[category] += 1
        for reason in reasons:
            reason_counts[reason] += 1
        cells[(identity["model"], identity["dataset"], identity["split"])][category] += 1
        ledger.append(
            {
                **identity,
                "category": category,
                "reasons": reasons,
                "artifact_paths": artifact_paths,
                "artifact_sha256": artifact_hashes,
                "locked_no_replay": category == "ambiguous_locked",
            }
        )

    terminal_count = sum(counts[name] for name in CATEGORIES if name != "missing_unreconciled")
    total = len(expected)
    status = "PASS"
    blockers: list[str] = []
    if total != int(freeze["prompt_ledger_rows"]) * len(freeze["models"]) * int(
        freeze["decoding"]["repeated_samples"]
    ):
        blockers.append("expected_request_count_does_not_match_freeze")
    if terminal_count != total or counts["missing_unreconciled"]:
        blockers.append("terminal_coverage_incomplete")
    if overlaps:
        blockers.append("artifact_category_overlap")
    if any(orphans.values()):
        blockers.append("orphan_artifacts")
    if provenance_violations:
        blockers.append("artifact_provenance_violation")
    if blockers:
        status = "FAIL"

    cell_rows = []
    for (model, dataset, split), row_counts in sorted(cells.items()):
        successful = row_counts["valid"] + row_counts["malformed"]
        included = row_counts["valid"]
        excluded_by_reason = {
            "malformed": row_counts["malformed"],
            "provider_failure": row_counts["provider_failure"],
            "locked_ambiguous": row_counts["ambiguous_locked"],
            "other_terminal": row_counts["other_terminal"],
            "missing_unreconciled": row_counts["missing_unreconciled"],
        }
        cell_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "split": split,
                "planned": sum(row_counts.values()),
                "successful": successful,
                "schema_valid": row_counts["valid"],
                "malformed": row_counts["malformed"],
                "provider_failure": row_counts["provider_failure"],
                "locked_ambiguous": row_counts["ambiguous_locked"],
                "included": included,
                "excluded": sum(excluded_by_reason.values()),
                "excluded_by_reason": excluded_by_reason,
                **{name: row_counts[name] for name in CATEGORIES},
            }
        )
    for row in cell_rows:
        if row["successful"] != row["schema_valid"] + row["malformed"]:
            blockers.append("cell_success_arithmetic_mismatch")
        if row["planned"] != row["included"] + row["excluded"]:
            blockers.append("cell_inclusion_arithmetic_mismatch")
    if blockers:
        status = "FAIL"
    ledger_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger).encode()
    manifest = {
        "schema": "trajuq_rq5_terminal_reconciliation_v1",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": freeze_sha,
        "planned_requests": total,
        "terminal_requests": terminal_count,
        "counts": {name: counts[name] for name in CATEGORIES},
        "reason_counts": dict(sorted(reason_counts.items())),
        "ambiguous_lock_count": counts["ambiguous_locked"],
        "ambiguous_lock_policy": "preserve_as_missing_never_resubmit",
        "explicit_ambiguous_registry_count": len(explicit_ambiguous),
        "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "cell_missingness": cell_rows,
        "orphan_counts": {name: len(values) for name, values in orphans.items()},
        "overlap_count": overlaps,
        "provenance_violation_count": provenance_violations,
        "blockers": blockers,
        "analysis_authorized": False,
        "analysis_authorization_note": "Terminal reconciliation does not replace registered scoring, aggregation, inference, or claim gates.",
    }
    return manifest, ledger


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile the frozen RQ5 terminal request universe without replay."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--freeze", default="results/rq5/TMLR_RQ5_BLACKBOX_FREEZE.json")
    parser.add_argument("--schema", default="configs/schemas/rq5_api_provenance_v1.json")
    parser.add_argument("--api-root", default="results/rq5/api")
    parser.add_argument(
        "--ambiguous-registry", default="configs/operations/rq5_ambiguous_uuid_registry.yaml"
    )
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--ledger-output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest, ledger = reconcile(
        root=root,
        freeze_path=root / args.freeze,
        schema_path=root / args.schema,
        api_root=root / args.api_root,
        ambiguous_registry_path=root / args.ambiguous_registry,
    )
    if not args.check_only:
        if not args.manifest_output or not args.ledger_output:
            raise SystemExit("materialization requires --manifest-output and --ledger-output")
        ledger_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger).encode()
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        write_new(args.ledger_output, ledger_bytes)
        write_new(args.manifest_output, manifest_bytes)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "status",
                    "planned_requests",
                    "terminal_requests",
                    "counts",
                    "ambiguous_lock_count",
                    "ledger_sha256",
                    "blockers",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    if manifest["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
