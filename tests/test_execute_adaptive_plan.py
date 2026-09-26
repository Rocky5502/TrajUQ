import hashlib
import json
from pathlib import Path

import scripts.execute_adaptive_plan as execute_adaptive_plan
from scripts.execute_adaptive_plan import (
    audit_passes,
    base_partition_state,
    evaluation_state,
    execution_snapshot,
    validate_plan_provenance,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_audited(path: Path, stage: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"example_id":"x"}\n', encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".audit.json").write_text(
        json.dumps(
            {"status": "PASS", "stage": stage, "sha256": digest}
        ),
        encoding="utf-8",
    )


def test_base_partition_requires_both_audited_files(tmp_path):
    raw = tmp_path / "base.raw.jsonl"
    scored = tmp_path / "base.scored.jsonl"
    spec = {"raw_output": str(raw), "scored_output": str(scored)}

    write_audited(raw, "raw")
    assert not base_partition_state(spec)["complete"]
    write_audited(scored, "scored")
    assert base_partition_state(spec)["complete"]


def test_adaptive_evaluation_rejects_partial_pair(tmp_path):
    result = tmp_path / "result.json"
    predictions = tmp_path / "predictions.jsonl"
    result.write_text("{}", encoding="utf-8")
    spec = {"result": str(result), "predictions": str(predictions)}
    assert evaluation_state(spec) == "PARTIAL"


def test_audit_stage_is_checked(tmp_path):
    raw = tmp_path / "base.raw.jsonl"
    write_audited(raw, "scored")
    assert not audit_passes(raw, "raw")


def test_execution_snapshot_requires_all_macros(tmp_path):
    raw_cal = tmp_path / "cal.raw.jsonl"
    scored_cal = tmp_path / "cal.scored.jsonl"
    raw_test = tmp_path / "test.raw.jsonl"
    scored_test = tmp_path / "test.scored.jsonl"
    for path, stage in (
        (raw_cal, "raw"),
        (scored_cal, "scored"),
        (raw_test, "raw"),
        (scored_test, "scored"),
    ):
        write_audited(path, stage)

    result = tmp_path / "cap.json"
    predictions = tmp_path / "cap.predictions.jsonl"
    result.write_text("{}", encoding="utf-8")
    predictions.write_text('{"example_id":"x"}\n', encoding="utf-8")

    macro = tmp_path / "macro.json"
    plan = {
        "expected_cells": 1,
        "fractions": [{"key": "cap_050", "value": 0.5, "primary": True}],
        "macro_aggregation": {
            "cap_050": {
                "output": str(macro),
                "argv": ["python", "aggregate.py"],
            }
        },
        "cells": [
            {
                "cell": "llada:gsm8k",
                "adapter": "llada",
                "dataset": "gsm8k",
                "calibration_base": {
                    "raw_output": str(raw_cal),
                    "scored_output": str(scored_cal),
                },
                "test_base": {
                    "raw_output": str(raw_test),
                    "scored_output": str(scored_test),
                },
                "evaluations": {
                    "cap_050": {
                        "result": str(result),
                        "predictions": str(predictions),
                    }
                },
            }
        ],
    }

    snapshot = execution_snapshot(plan)
    assert snapshot["complete_cells"] == 1
    assert snapshot["status"] == "IN_PROGRESS"

    macro.write_text(
        json.dumps(
            {
                "cell_set_complete": True,
                "observed_cells": 1,
                "fraction_key": "cap_050",
            }
        ),
        encoding="utf-8",
    )
    assert execution_snapshot(plan)["status"] == "COMPLETE"


def test_adaptive_provenance_rejects_refined_input_drift(
    tmp_path,
    monkeypatch,
):
    commit = "c" * 40
    primary_plan = tmp_path / "primary_plan.json"
    primary_config = tmp_path / "primary.yaml"
    generation_input = tmp_path / "calibration.jsonl"
    refined = tmp_path / "refined.scored.jsonl"
    refined_audit = tmp_path / "refined.scored.jsonl.audit.json"

    primary_plan.write_text('{"status":"PASS"}\n', encoding="utf-8")
    primary_config.write_text("experiment:\n  seed: 2027\n", encoding="utf-8")
    generation_input.write_text('{"example_id":"x"}\n', encoding="utf-8")
    refined.write_text('{"example_id":"x"}\n', encoding="utf-8")
    refined_audit.write_text(
        json.dumps(
            {
                "status": "PASS",
                "stage": "scored",
                "sha256": _sha256(refined),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    partition = {
        "generation_input": str(generation_input),
        "generation_input_sha256": _sha256(generation_input),
        "refined_scored": str(refined),
        "refined_scored_sha256": _sha256(refined),
        "refined_audit": str(refined_audit),
        "refined_audit_sha256": _sha256(refined_audit),
    }
    plan = {
        "protocol": "trajuq_adaptive_matrix_plan_v3",
        "status": "PASS",
        "experiment_git_commit": commit,
        "primary_plan": str(primary_plan),
        "primary_plan_sha256": _sha256(primary_plan),
        "primary_config": str(primary_config),
        "primary_config_sha256": _sha256(primary_config),
        "cells": [
            {
                "cell": "llada:gsm8k",
                "provenance": {
                    "calibration": dict(partition),
                    "test": dict(partition),
                },
            }
        ],
    }
    monkeypatch.setattr(execute_adaptive_plan, "git_head", lambda: commit)

    assert validate_plan_provenance(plan) == []

    refined.write_text('{"example_id":"tampered"}\n', encoding="utf-8")
    failures = validate_plan_provenance(plan)
    assert any("refined scored input hash drifted" in item for item in failures)
    assert any("refined audit no longer binds input" in item for item in failures)
