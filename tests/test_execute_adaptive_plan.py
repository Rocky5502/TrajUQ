import hashlib
import json
from pathlib import Path

from scripts.execute_adaptive_plan import (
    audit_passes,
    base_partition_state,
    evaluation_state,
    execution_snapshot,
)


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
