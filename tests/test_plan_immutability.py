import json
from pathlib import Path

import pytest

from trajuq.io import write_json_immutable


def test_write_json_immutable_creates_and_preserves_identical_payload(tmp_path):
    path = tmp_path / "plan.json"
    payload = {"protocol": "example_v1", "status": "PASS", "value": 1}

    assert write_json_immutable(path, payload) == "CREATED"
    first = path.read_bytes()
    assert write_json_immutable(path, payload) == "PRESERVED"
    assert path.read_bytes() == first
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_write_json_immutable_rejects_different_existing_payload(tmp_path):
    path = tmp_path / "plan.json"
    write_json_immutable(path, {"value": 1})

    with pytest.raises(FileExistsError, match="Refusing to overwrite immutable JSON artifact"):
        write_json_immutable(path, {"value": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_all_frozen_planners_use_immutable_json_writer():
    planners = [
        "scripts/plan_primary_matrix.py",
        "scripts/plan_primary_analysis.py",
        "scripts/plan_repeated_sampling.py",
        "scripts/plan_adaptive_runs.py",
    ]
    for planner in planners:
        text = Path(planner).read_text(encoding="utf-8")
        assert "write_json_immutable" in text, planner
        assert "out.write_text(" not in text, planner


def test_primary_readiness_uses_immutable_json_writer():
    text = Path("scripts/primary_readiness.py").read_text(encoding="utf-8")
    assert "write_json_immutable(out, payload)" in text
    assert "out.write_text(" not in text
