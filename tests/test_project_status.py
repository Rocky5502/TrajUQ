from pathlib import Path

import yaml

from scripts.project_status import build_snapshot, dotted_get, render_markdown


def test_dotted_get_reads_nested_key():
    assert dotted_get({"a": {"b": 3}}, "a.b") == 3


def test_snapshot_separates_implementation_from_pending_execution(tmp_path):
    evidence = tmp_path / "implemented.txt"
    evidence.write_text("ok", encoding="utf-8")

    config = {
        "project": {"target": "test"},
        "implementation": [
            {
                "id": "impl",
                "label": "Implementation",
                "state": "DONE",
                "evidence": ["implemented.txt"],
            }
        ],
        "execution_gates": [
            {
                "id": "gate",
                "label": "Execution gate",
                "stage": "pre_primary",
                "artifact": "results/gate.json",
                "expect": {"key": "status", "equals": "PASS"},
            }
        ],
    }

    snapshot = build_snapshot(tmp_path, config)
    assert snapshot["summary"]["implementation_verified"] == 1
    assert snapshot["summary"]["execution_passed"] == 0
    assert snapshot["summary"]["execution_pending"] == 1
    assert snapshot["execution_gates"][0]["state"] == "PENDING"


def test_present_failed_gate_is_not_counted_as_pending(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "gate.json").write_text(
        '{"status": "FAIL"}',
        encoding="utf-8",
    )
    config = {
        "project": {"target": "test"},
        "implementation": [],
        "execution_gates": [
            {
                "id": "gate",
                "label": "Gate",
                "artifact": "results/gate.json",
                "expect": {"key": "status", "equals": "PASS"},
            }
        ],
    }
    snapshot = build_snapshot(tmp_path, config)
    assert snapshot["execution_gates"][0]["state"] == "FAIL"
    assert snapshot["summary"]["execution_failed_or_invalid"] == 1
    assert snapshot["summary"]["execution_pending"] == 0
    assert "Execution gates passed" in render_markdown(snapshot)


def test_project_status_config_loads():
    config = yaml.safe_load(
        Path("configs/project_status.yaml").read_text(encoding="utf-8")
    )
    assert config["project"]["primary_cells"] == 18
    assert config["project"]["adaptive_primary_cap"] == 0.50
