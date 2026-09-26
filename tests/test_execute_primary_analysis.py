import hashlib
import json
from pathlib import Path

import scripts.execute_primary_analysis as execute_primary_analysis
from scripts.execute_primary_analysis import (
    execution_snapshot,
    output_valid,
    step_state,
    validate_plan_provenance,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_output_valid_json_and_jsonl(tmp_path):
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    metrics.write_text('{"auroc": 0.5}', encoding="utf-8")
    predictions.write_text('{"example_id":"x"}\n', encoding="utf-8")
    assert output_valid(metrics)
    assert output_valid(predictions)


def test_evaluate_step_detects_partial_outputs(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"auroc": 0.5}', encoding="utf-8")
    cell = {
        "outputs": {
            "metrics": str(metrics),
            "predictions": str(tmp_path / "predictions.jsonl"),
        }
    }
    assert step_state(cell, ("metrics", "predictions")) == "PARTIAL"


def test_analysis_snapshot_requires_macro_and_claim_gate(tmp_path):
    outputs = {
        "metrics": tmp_path / "metrics.json",
        "predictions": tmp_path / "predictions.jsonl",
        "nested_bootstrap": tmp_path / "nested.json",
        "feature_ablation": tmp_path / "ablation.json",
        "raw_risk_diagnostic": tmp_path / "raw.json",
    }
    for key, path in outputs.items():
        if key == "predictions":
            path.write_text('{"example_id":"x"}\n', encoding="utf-8")
        else:
            path.write_text("{}", encoding="utf-8")

    macro = tmp_path / "macro.json"
    claim = tmp_path / "claim.json"
    plan = {
        "status": "PASS",
        "cell_count": 1,
        "cell_analyses": [
            {
                "cell": "llada:gsm8k",
                "adapter": "llada",
                "dataset": "gsm8k",
                "outputs": {key: str(value) for key, value in outputs.items()},
            }
        ],
        "final_outputs": {
            "macro_primary": str(macro),
            "rq1_claim_gate": str(claim),
        },
    }

    snapshot = execution_snapshot(plan)
    assert snapshot["complete_cells"] == 1
    assert snapshot["status"] == "IN_PROGRESS"

    macro.write_text(
        json.dumps({"cell_set_complete": True}),
        encoding="utf-8",
    )
    claim.write_text(
        json.dumps({"decision": "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM"}),
        encoding="utf-8",
    )
    snapshot = execution_snapshot(plan)
    assert snapshot["status"] == "COMPLETE"


def test_analysis_provenance_rejects_scored_input_drift(
    tmp_path,
    monkeypatch,
):
    commit = "a" * 40
    primary_plan = tmp_path / "primary_plan.json"
    primary_config = tmp_path / "primary.yaml"
    scored = tmp_path / "calibration.scored.jsonl"
    audit = tmp_path / "calibration.scored.jsonl.audit.json"

    primary_plan.write_text('{"status":"PASS"}\n', encoding="utf-8")
    primary_config.write_text("experiment:\n  seed: 2027\n", encoding="utf-8")
    scored.write_text('{"example_id":"x"}\n', encoding="utf-8")
    audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": commit,
                "git_clean": True,
                "stage": "scored",
                "sha256": _sha256(scored),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    input_spec = {
        "path": str(scored),
        "sha256": _sha256(scored),
        "audit_path": str(audit),
        "audit_sha256": _sha256(audit),
    }
    plan = {
        "protocol": "trajuq_primary_analysis_plan_v2",
        "status": "PASS",
        "experiment_git_commit": commit,
        "primary_plan": str(primary_plan),
        "primary_plan_sha256": _sha256(primary_plan),
        "primary_config": str(primary_config),
        "primary_config_sha256": _sha256(primary_config),
        "cell_analyses": [
            {
                "cell": "llada:gsm8k",
                "inputs": {
                    "calibration": dict(input_spec),
                    "test": dict(input_spec),
                },
            }
        ],
    }
    monkeypatch.setattr(execute_primary_analysis, "git_head", lambda: commit)
    monkeypatch.setattr(
        execute_primary_analysis,
        "git_status_porcelain",
        lambda: "",
    )

    assert validate_plan_provenance(plan) == []

    scored.write_text('{"example_id":"tampered"}\n', encoding="utf-8")
    failures = validate_plan_provenance(plan)
    assert any("scored input hash differs" in item for item in failures)
    assert any("audit no longer binds scored input" in item for item in failures)


def test_analysis_provenance_rejects_audit_from_other_commit(
    tmp_path,
    monkeypatch,
):
    commit = "a" * 40
    primary_plan = tmp_path / "primary_plan.json"
    primary_config = tmp_path / "primary.yaml"
    scored = tmp_path / "test.scored.jsonl"
    audit = tmp_path / "test.scored.jsonl.audit.json"

    primary_plan.write_text('{"status":"PASS"}\n', encoding="utf-8")
    primary_config.write_text("experiment:\n  seed: 2027\n", encoding="utf-8")
    scored.write_text('{"example_id":"x"}\n', encoding="utf-8")
    audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": "other-commit",
                "git_clean": True,
                "stage": "scored",
                "sha256": _sha256(scored),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = {
        "path": str(scored),
        "sha256": _sha256(scored),
        "audit_path": str(audit),
        "audit_sha256": _sha256(audit),
    }
    plan = {
        "protocol": "trajuq_primary_analysis_plan_v2",
        "status": "PASS",
        "experiment_git_commit": commit,
        "primary_plan": str(primary_plan),
        "primary_plan_sha256": _sha256(primary_plan),
        "primary_config": str(primary_config),
        "primary_config_sha256": _sha256(primary_config),
        "cell_analyses": [
            {
                "cell": "llada:gsm8k",
                "inputs": {"calibration": dict(spec), "test": dict(spec)},
            }
        ],
    }
    monkeypatch.setattr(execute_primary_analysis, "git_head", lambda: commit)
    monkeypatch.setattr(execute_primary_analysis, "git_status_porcelain", lambda: "")

    failures = validate_plan_provenance(plan)
    assert any("clean frozen scored PASS" in item for item in failures)
