import hashlib
import json
from pathlib import Path

from scripts.execute_primary_plan import (
    audit_passes,
    execution_snapshot,
    partition_state,
    validate_plan_provenance,
)


def write_artifact(path: Path, content: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_pass_audit(
    path: Path,
    *,
    stage: str | None = None,
    commit: str = "frozen-commit",
) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = path.with_suffix(path.suffix + ".audit.json")
    resolved_stage = stage or ("scored" if "scored" in path.name else "raw")
    audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": commit,
                "git_clean": True,
                "stage": resolved_stage,
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )


def test_audit_passes_requires_matching_hash(tmp_path):
    artifact = tmp_path / "run.jsonl"
    write_artifact(artifact)
    write_pass_audit(artifact)
    assert audit_passes(artifact, "raw")

    artifact.write_text("changed\n", encoding="utf-8")
    assert not audit_passes(artifact, "raw")


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


def test_validate_plan_provenance_accepts_bound_inputs(tmp_path, monkeypatch):
    primary = tmp_path / "primary.yaml"
    readiness = tmp_path / "readiness.json"
    model = tmp_path / "llada.yaml"
    cal = tmp_path / "cal.jsonl"
    test = tmp_path / "test.jsonl"
    for path, text in (
        (primary, "experiment: {}\n"),
        (readiness, '{"status":"PASS"}\n'),
        (model, "revision: abc\n"),
        (cal, '{"id":"c"}\n'),
        (test, '{"id":"t"}\n'),
    ):
        path.write_text(text, encoding="utf-8")

    monkeypatch.setattr(
        "scripts.execute_primary_plan.git_head",
        lambda: "frozen-commit",
    )
    monkeypatch.setattr(
        "scripts.execute_primary_plan.git_status_porcelain",
        lambda: "",
    )
    plan = {
        "protocol": "trajuq_primary_matrix_plan_v2",
        "experiment_git_commit": "frozen-commit",
        "primary_config": str(primary),
        "primary_config_sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
        "readiness": str(readiness),
        "readiness_sha256": hashlib.sha256(readiness.read_bytes()).hexdigest(),
        "model_configs": {
            "llada": {
                "path": str(model),
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        },
        "cells": [
            {
                "cell": "llada:gsm8k",
                "calibration": {
                    "input": str(cal),
                    "input_sha256": hashlib.sha256(cal.read_bytes()).hexdigest(),
                },
                "test": {
                    "input": str(test),
                    "input_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
                },
            }
        ],
    }

    assert validate_plan_provenance(plan) == []

    monkeypatch.setattr(
        "scripts.execute_primary_plan.git_status_porcelain",
        lambda: " M scripts/run_generation.py",
    )
    failures = validate_plan_provenance(plan)
    assert any("worktree is dirty" in item for item in failures)


def test_validate_plan_provenance_rejects_mutated_input(tmp_path, monkeypatch):
    primary = tmp_path / "primary.yaml"
    readiness = tmp_path / "readiness.json"
    model = tmp_path / "llada.yaml"
    cal = tmp_path / "cal.jsonl"
    test = tmp_path / "test.jsonl"
    for path in (primary, readiness, model, cal, test):
        path.write_text("frozen\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.execute_primary_plan.git_head",
        lambda: "frozen-commit",
    )
    cal_hash = hashlib.sha256(cal.read_bytes()).hexdigest()
    plan = {
        "protocol": "trajuq_primary_matrix_plan_v2",
        "experiment_git_commit": "frozen-commit",
        "primary_config": str(primary),
        "primary_config_sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
        "readiness": str(readiness),
        "readiness_sha256": hashlib.sha256(readiness.read_bytes()).hexdigest(),
        "model_configs": {
            "llada": {
                "path": str(model),
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        },
        "cells": [
            {
                "cell": "llada:gsm8k",
                "calibration": {
                    "input": str(cal),
                    "input_sha256": cal_hash,
                },
                "test": {
                    "input": str(test),
                    "input_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
                },
            }
        ],
    }
    cal.write_text("mutated\n", encoding="utf-8")

    failures = validate_plan_provenance(plan)
    assert any("input hash differs from plan" in item for item in failures)


def test_audit_passes_rejects_other_commit(tmp_path):
    artifact = tmp_path / "run.raw.jsonl"
    write_artifact(artifact)
    write_pass_audit(artifact, stage="raw", commit="other-commit")
    assert not audit_passes(artifact, "raw", "frozen-commit")
