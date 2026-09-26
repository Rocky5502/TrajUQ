import hashlib
import json
from pathlib import Path

import scripts.execute_repeated_sampling as execute_repeated_sampling
from scripts.execute_repeated_sampling import (
    audit_passes,
    execution_snapshot,
    generation_state,
    pair_state,
    validate_plan_provenance,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_raw(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"example_id":"x"}\n', encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".audit.json").write_text(
        json.dumps(
            {"status": "PASS", "stage": "raw", "sha256": digest}
        ),
        encoding="utf-8",
    )


def test_generation_state_requires_every_seed_and_manifest(tmp_path):
    argv = [
        "python",
        "scripts/run_repeated_samples.py",
        "--output-dir",
        str(tmp_path),
    ]
    write_raw(tmp_path / "seed-2027.raw.jsonl")
    write_raw(tmp_path / "seed-2028.raw.jsonl")
    assert not generation_state(argv, [2027, 2028])["complete"]

    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    assert generation_state(argv, [2027, 2028])["complete"]


def test_pair_state_rejects_partial_outputs(tmp_path):
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    metrics.write_text("{}", encoding="utf-8")
    assert pair_state(metrics, predictions) == "PARTIAL"


def test_audit_hash_must_match(tmp_path):
    path = tmp_path / "seed.raw.jsonl"
    write_raw(path)
    assert audit_passes(path)
    path.write_text("changed\n", encoding="utf-8")
    assert not audit_passes(path)


def test_execution_snapshot_waits_for_macro(tmp_path):
    cal = tmp_path / "cal"
    test = tmp_path / "test"
    for root in (cal, test):
        write_raw(root / "seed-2027.raw.jsonl")
        (root / "manifest.json").write_text("{}", encoding="utf-8")

    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    comparison = tmp_path / "comparison.json"
    metrics.write_text("{}", encoding="utf-8")
    predictions.write_text('{"example_id":"x"}\n', encoding="utf-8")
    comparison.write_text("{}", encoding="utf-8")

    macro = tmp_path / "macro.json"
    plan = {
        "expected_cells": 1,
        "macro_output": str(macro),
        "cells": [
            {
                "cell": "llada:gsm8k",
                "adapter": "llada",
                "dataset": "gsm8k",
                "seeds": [2027],
                "calibration_generation_argv": [
                    "python",
                    "x",
                    "--output-dir",
                    str(cal),
                ],
                "test_generation_argv": [
                    "python",
                    "x",
                    "--output-dir",
                    str(test),
                ],
                "repeated_metrics": str(metrics),
                "repeated_predictions": str(predictions),
                "direct_comparison": str(comparison),
            }
        ],
    }
    snapshot = execution_snapshot(plan)
    assert snapshot["complete_cells"] == 1
    assert snapshot["status"] == "IN_PROGRESS"

    macro.write_text(
        json.dumps(
            {"cell_set_complete": True, "observed_cells": 1}
        ),
        encoding="utf-8",
    )
    assert execution_snapshot(plan)["status"] == "COMPLETE"


def test_repeated_provenance_rejects_primary_prediction_drift(
    tmp_path,
    monkeypatch,
):
    commit = "b" * 40
    primary_plan = tmp_path / "primary_plan.json"
    analysis_plan = tmp_path / "analysis_plan.json"
    primary_config = tmp_path / "primary.yaml"
    generation_input = tmp_path / "calibration.jsonl"
    scored = tmp_path / "calibration.scored.jsonl"
    scored_audit = tmp_path / "calibration.scored.jsonl.audit.json"
    predictions = tmp_path / "trajuq_predictions.jsonl"

    primary_plan.write_text('{"status":"PASS"}\n', encoding="utf-8")
    analysis_plan.write_text('{"status":"PASS"}\n', encoding="utf-8")
    primary_config.write_text("experiment:\n  seed: 2027\n", encoding="utf-8")
    generation_input.write_text('{"example_id":"x"}\n', encoding="utf-8")
    scored.write_text('{"example_id":"x"}\n', encoding="utf-8")
    scored_audit.write_text(
        json.dumps(
            {
                "status": "PASS",
                "stage": "scored",
                "sha256": _sha256(scored),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    predictions.write_text(
        '{"example_id":"x","risk":0.5}\n',
        encoding="utf-8",
    )

    partition = {
        "generation_input": str(generation_input),
        "generation_input_sha256": _sha256(generation_input),
        "scored_target": str(scored),
        "scored_target_sha256": _sha256(scored),
        "scored_audit": str(scored_audit),
        "scored_audit_sha256": _sha256(scored_audit),
    }
    plan = {
        "protocol": "trajuq_repeated_sampling_plan_v2",
        "status": "PASS",
        "experiment_git_commit": commit,
        "primary_plan": str(primary_plan),
        "primary_plan_sha256": _sha256(primary_plan),
        "primary_analysis_plan": str(analysis_plan),
        "primary_analysis_plan_sha256": _sha256(analysis_plan),
        "primary_config": str(primary_config),
        "primary_config_sha256": _sha256(primary_config),
        "cells": [
            {
                "cell": "llada:gsm8k",
                "provenance": {
                    "calibration": dict(partition),
                    "test": dict(partition),
                    "trajuq_predictions": str(predictions),
                    "trajuq_predictions_sha256": _sha256(predictions),
                },
            }
        ],
    }
    monkeypatch.setattr(execute_repeated_sampling, "git_head", lambda: commit)
    monkeypatch.setattr(
        execute_repeated_sampling,
        "git_status_porcelain",
        lambda: "",
    )

    assert validate_plan_provenance(plan) == []

    predictions.write_text(
        '{"example_id":"x","risk":0.9}\n',
        encoding="utf-8",
    )
    failures = validate_plan_provenance(plan)
    assert any("TrajUQ predictions hash drifted" in item for item in failures)
