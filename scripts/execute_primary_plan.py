#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trajuq.io import read_jsonl


STEP_KEYS = (
    ("generate", "generate_argv"),
    ("audit_raw", "audit_raw_argv"),
    ("score", "score_argv"),
    ("audit_scored", "audit_scored_argv"),
)


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


def validate_readiness_bound_sources(readiness_path: Path) -> list[str]:
    failures: list[str] = []
    if not readiness_path.is_file():
        return [f"primary readiness artifact is missing: {readiness_path}"]
    try:
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"primary readiness artifact is invalid: {exc}"]

    bindings = readiness.get("source_bindings")
    if not isinstance(bindings, list) or not bindings:
        return ["primary readiness has no source-binding manifest"]

    seen: set[str] = set()
    for item in bindings:
        if not isinstance(item, dict):
            failures.append("primary readiness source binding is malformed")
            continue
        path_text = str(item.get("path", "")).strip()
        expected_sha = str(item.get("sha256", "")).strip()
        if not path_text or not expected_sha:
            failures.append("primary readiness source binding lacks path/SHA-256")
            continue
        if path_text in seen:
            failures.append(
                f"primary readiness source binding duplicates path: {path_text}"
            )
            continue
        seen.add(path_text)
        path = Path(path_text)
        if not path.is_file():
            failures.append(
                f"primary readiness bound source is missing: {path_text}"
            )
            continue
        if sha256_file(path) != expected_sha:
            failures.append(
                f"primary readiness bound source hash drifted: {path_text}"
            )
    return failures


def validate_plan_provenance(
    plan: dict[str, Any],
    *,
    require_current_commit: bool = True,
) -> list[str]:
    failures: list[str] = []
    if plan.get("protocol") != "trajuq_primary_matrix_plan_v2":
        failures.append("primary plan protocol is not v2")

    expected_commit = plan.get("experiment_git_commit")
    if require_current_commit:
        current_head = git_head()
        if not current_head:
            failures.append("current git commit could not be resolved")
        elif current_head != expected_commit:
            failures.append(
                "current git commit differs from the primary plan experiment commit"
            )
        current_status = git_status_porcelain()
        if current_status is None:
            failures.append("current git worktree status could not be resolved")
        elif current_status:
            failures.append(
                "current git worktree is dirty; frozen execution requires a clean checkout"
            )

    for key, hash_key in (
        ("primary_config", "primary_config_sha256"),
        ("readiness", "readiness_sha256"),
        ("revision_ledger", "revision_ledger_sha256"),
    ):
        path = Path(str(plan.get(key, "")))
        expected_hash = plan.get(hash_key)
        if not path.exists() or not expected_hash:
            failures.append(f"primary plan missing bound {key} artifact")
        elif sha256_file(path) != expected_hash:
            failures.append(f"{key} hash differs from primary plan")

    readiness_path = Path(str(plan.get("readiness", "")))
    failures.extend(validate_readiness_bound_sources(readiness_path))

    for adapter, spec in plan.get("model_configs", {}).items():
        path = Path(str(spec.get("path", "")))
        expected_hash = spec.get("sha256")
        if not path.exists() or not expected_hash:
            failures.append(f"{adapter}: bound frozen model config is missing")
        elif sha256_file(path) != expected_hash:
            failures.append(f"{adapter}: frozen model config hash differs from plan")

    for cell in plan.get("cells", []):
        label = str(cell.get("cell", "<unknown>"))
        for key, hash_key in (
            ("split_manifest", "split_manifest_sha256"),
            ("split_audit", "split_audit_sha256"),
        ):
            path = Path(str(cell.get(key, "")))
            expected_hash = cell.get(hash_key)
            if not path.exists() or not expected_hash:
                failures.append(f"{label}: bound {key} is missing")
            elif sha256_file(path) != expected_hash:
                failures.append(f"{label}: {key} hash differs from plan")

        for partition in ("calibration", "test"):
            spec = cell.get(partition, {})
            path = Path(str(spec.get("input", "")))
            expected_hash = spec.get("input_sha256")
            if not path.exists() or not expected_hash:
                failures.append(
                    f"{label} {partition}: bound input is missing"
                )
            elif sha256_file(path) != expected_hash:
                failures.append(
                    f"{label} {partition}: input hash differs from plan"
                )
    return failures


def audit_path(artifact: Path) -> Path:
    return artifact.with_suffix(artifact.suffix + ".audit.json")


def expected_input_ids(spec: dict[str, Any]) -> list[str]:
    path = Path(str(spec["input"]))
    rows = list(read_jsonl(path))
    ids = [str(row.get("id", "")) for row in rows]
    if not rows:
        raise ValueError(f"planned input is empty: {path}")
    if any(not item_id for item_id in ids):
        raise ValueError(f"planned input contains a row without id: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"planned input contains duplicate ids: {path}")
    return ids


def artifact_example_ids(path: Path) -> list[str]:
    rows = list(read_jsonl(path))
    ids = [str(row.get("example_id", "")) for row in rows]
    if any(not item_id for item_id in ids):
        raise ValueError(f"artifact contains a row without example_id: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"artifact contains duplicate example_ids: {path}")
    return ids


def artifact_id_state(path: Path, spec: dict[str, Any]) -> str:
    if not path.exists():
        return "MISSING"
    expected = expected_input_ids(spec)
    try:
        observed = artifact_example_ids(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return "INVALID"
    if observed == expected:
        return "COMPLETE"
    if len(observed) < len(expected) and observed == expected[: len(observed)]:
        return "PREFIX"
    return "INVALID"


def audit_passes(
    artifact: Path,
    expected_stage: str,
    expected_commit: str | None = None,
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
        and payload.get("stage") == expected_stage
        and str(payload.get("sha256", "")) == sha256_file(artifact)
        and (
            expected_commit is None
            or payload.get("git_commit") == expected_commit
        )
    )


def partition_state(
    spec: dict[str, Any],
    expected_commit: str | None = None,
) -> dict[str, Any]:
    raw = Path(str(spec["raw_output"]))
    scored = Path(str(spec["scored_output"]))
    raw_exists = raw.exists()
    scored_exists = scored.exists()
    raw_id_state = artifact_id_state(raw, spec)
    scored_id_state = artifact_id_state(scored, spec)
    raw_audit_pass = audit_passes(raw, "raw", expected_commit)
    scored_audit_pass = audit_passes(scored, "scored", expected_commit)
    return {
        "raw_output": str(raw),
        "scored_output": str(scored),
        "raw_exists": raw_exists,
        "raw_id_state": raw_id_state,
        "raw_audit_pass": raw_audit_pass,
        "scored_exists": scored_exists,
        "scored_id_state": scored_id_state,
        "scored_audit_pass": scored_audit_pass,
        "complete": bool(
            raw_id_state == "COMPLETE"
            and raw_audit_pass
            and scored_id_state == "COMPLETE"
            and scored_audit_pass
        ),
    }


def execution_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    completed_partitions = 0
    total_partitions = 0

    expected_commit = str(plan.get("experiment_git_commit", "")) or None
    for cell in plan.get("cells", []):
        partitions: dict[str, Any] = {}
        for name in ("calibration", "test"):
            total_partitions += 1
            state = partition_state(cell[name], expected_commit)
            completed_partitions += int(state["complete"])
            partitions[name] = state
        cells.append(
            {
                "cell": cell["cell"],
                "adapter": cell["adapter"],
                "dataset": cell["dataset"],
                "partitions": partitions,
                "complete": all(
                    partition["complete"]
                    for partition in partitions.values()
                ),
            }
        )

    complete_cells = sum(cell["complete"] for cell in cells)
    expected_cells = int(plan.get("expected_cells", len(cells)))
    status = (
        "COMPLETE"
        if len(cells) == expected_cells
        and complete_cells == expected_cells
        and completed_partitions == total_partitions
        else "IN_PROGRESS"
    )
    return {
        "protocol": "trajuq_primary_execution_state_v1",
        "status": status,
        "plan_protocol": plan.get("protocol"),
        "experiment_git_commit": plan.get("experiment_git_commit"),
        "expected_cells": expected_cells,
        "observed_cells": len(cells),
        "complete_cells": complete_cells,
        "total_partitions": total_partitions,
        "complete_partitions": completed_partitions,
        "partition_fraction": (
            completed_partitions / total_partitions
            if total_partitions
            else 0.0
        ),
        "cells": cells,
    }


def write_snapshot(plan: dict[str, Any], path: Path) -> dict[str, Any]:
    snapshot = execution_snapshot(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot


def command_needed(
    step: str,
    spec: dict[str, Any],
    expected_commit: str | None = None,
) -> bool:
    raw = Path(str(spec["raw_output"]))
    scored = Path(str(spec["scored_output"]))
    if step == "generate":
        state = artifact_id_state(raw, spec)
        if state == "INVALID":
            raise ValueError(
                f"raw artifact is not an exact prefix of the frozen input: {raw}"
            )
        return state != "COMPLETE"
    if step == "audit_raw":
        return not audit_passes(raw, "raw", expected_commit)
    if step == "score":
        state = artifact_id_state(scored, spec)
        if state in {"PREFIX", "INVALID"}:
            raise ValueError(
                f"partial/invalid scored artifact requires inspection before rerun: {scored}"
            )
        return state == "MISSING"
    if step == "audit_scored":
        return not audit_passes(scored, "scored", expected_commit)
    raise ValueError(step)


def append_event(
    path: Path,
    *,
    cell: str,
    partition: str,
    step: str,
    argv: list[str],
    event: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cell": cell,
        "partition": partition,
        "step": step,
        "event": event,
        "argv": argv,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Safely inspect or serially execute the frozen TrajUQ primary plan. "
            "Existing artifacts are never overwritten."
        )
    )
    parser.add_argument(
        "--plan",
        default="results/primary_matrix_plan.json",
    )
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        help="Optional exact cell label (for example llada:gsm8k); repeatable.",
    )
    parser.add_argument(
        "--partition",
        choices=["calibration", "test", "both"],
        default="both",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run missing frozen commands; default is dry-run/status only.",
    )
    parser.add_argument(
        "--state-output",
        default="results/primary_execution_state.json",
    )
    parser.add_argument(
        "--event-log",
        default="results/primary_execution_events.jsonl",
    )
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.exists():
        raise SystemExit(f"Missing primary plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "PASS":
        raise SystemExit("Primary matrix plan is not PASS.")

    provenance_failures = validate_plan_provenance(plan)
    if provenance_failures:
        raise SystemExit(
            "Primary plan provenance check failed:\n- "
            + "\n- ".join(provenance_failures)
        )

    expected = int(plan.get("expected_cells", 0))
    cells = list(plan.get("cells", []))
    if expected <= 0 or len(cells) != expected:
        raise SystemExit(
            f"Primary plan cell count mismatch: {len(cells)} != {expected}"
        )

    requested = set(args.cell)
    known = {str(cell["cell"]) for cell in cells}
    unknown = sorted(requested - known)
    if unknown:
        raise SystemExit(f"Unknown requested cells: {unknown}")

    selected = [
        cell for cell in cells
        if not requested or str(cell["cell"]) in requested
    ]
    expected_commit = str(plan.get("experiment_git_commit", "")) or None
    partitions = (
        ("calibration", "test")
        if args.partition == "both"
        else (args.partition,)
    )

    state_path = Path(args.state_output)
    event_path = Path(args.event_log)
    initial = write_snapshot(plan, state_path)
    print(
        "[TrajUQ] primary progress: "
        f"{initial['complete_cells']}/{initial['expected_cells']} cells, "
        f"{initial['complete_partitions']}/{initial['total_partitions']} partitions"
    )

    for cell in selected:
        cell_name = str(cell["cell"])
        for partition in partitions:
            spec = cell[partition]
            for step, key in STEP_KEYS:
                argv = [str(value) for value in spec[key]]
                needed = command_needed(step, spec, expected_commit)
                if not needed:
                    print(
                        f"[SKIP] {cell_name} {partition} {step}: "
                        "validated artifact already present"
                    )
                    continue

                print(
                    f"[{'RUN' if args.execute else 'DRY'}] "
                    f"{cell_name} {partition} {step}: "
                    + " ".join(argv)
                )
                if not args.execute:
                    continue

                append_event(
                    event_path,
                    cell=cell_name,
                    partition=partition,
                    step=step,
                    argv=argv,
                    event="START",
                )
                try:
                    subprocess.run(argv, check=True)
                except subprocess.CalledProcessError:
                    append_event(
                        event_path,
                        cell=cell_name,
                        partition=partition,
                        step=step,
                        argv=argv,
                        event="FAIL",
                    )
                    write_snapshot(plan, state_path)
                    raise
                append_event(
                    event_path,
                    cell=cell_name,
                    partition=partition,
                    step=step,
                    argv=argv,
                    event="PASS",
                )
                write_snapshot(plan, state_path)

    final = write_snapshot(plan, state_path)
    print(json.dumps(final["status"], indent=2))
    print(
        "[TrajUQ] final primary progress: "
        f"{final['complete_cells']}/{final['expected_cells']} cells, "
        f"{final['complete_partitions']}/{final['total_partitions']} partitions"
    )


if __name__ == "__main__":
    main()
