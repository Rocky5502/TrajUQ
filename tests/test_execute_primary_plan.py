import hashlib
import json
from pathlib import Path

from scripts.execute_primary_plan import (
    audit_passes,
    execution_snapshot,
    partition_state,
)


def write_artifact(path: Path, content: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_pass_audit(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = path.with_suffix(path.suffix + ".audit.json")
    audit.write_text(
        json.dumps({"status": "PASS", "sha256": digest}),
        encoding="utf-8",
    )


def test_audit_passes_requires_matching_hash(tmp_path):
    artifact = tmp_path / "run.jsonl"
    write_artifact(artifact)
    write_pass_audit(artifact)
    assert audit_passes(artifact)

    artifact.write_text("changed\n", encoding="utf-8")
    assert not audit_passes(artifact)


def test_partition_complete_only_after_both_audited_artifacts(tmp_path):
    raw = tmp_path / "cal.raw.jsonl"
    scored = tmp_path / "cal.scored.jsonl"
    spec = {"raw_output": str(raw), "scored_output": str(scored)}

    assert not partition_state(spec)["complete"]
    write_artifact(raw)
    write_pass_audit(raw)
    assert not partition_state(spec)["complete"]
    write_artifact(scored)
    write_pass_audit(scored)
    assert partition_state(spec)["complete"]


def test_execution_snapshot_counts_complete_partitions(tmp_path):
    cal_raw = tmp_path / "cal.raw.jsonl"
    cal_scored = tmp_path / "cal.scored.jsonl"
    for path in (cal_raw, cal_scored):
        write_artifact(path)
        write_pass_audit(path)

    plan = {
        "protocol": "trajuq_primary_matrix_plan_v1",
        "expected_cells": 1,
        "cells": [
            {
                "cell": "llada:gsm8k",
                "adapter": "llada",
                "dataset": "gsm8k",
                "calibration": {
                    "raw_output": str(cal_raw),
                    "scored_output": str(cal_scored),
                },
                "test": {
                    "raw_output": str(tmp_path / "test.raw.jsonl"),
                    "scored_output": str(tmp_path / "test.scored.jsonl"),
                },
            }
        ],
    }
    snapshot = execution_snapshot(plan)
    assert snapshot["complete_cells"] == 0
    assert snapshot["complete_partitions"] == 1
    assert snapshot["total_partitions"] == 2
    assert snapshot["status"] == "IN_PROGRESS"
