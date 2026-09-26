import json
from pathlib import Path

from scripts.execute_primary_analysis import (
    execution_snapshot,
    output_valid,
    step_state,
)


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
