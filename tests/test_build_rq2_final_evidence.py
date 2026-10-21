import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_rq2_final_evidence import EXPECTED_COMMIT, build


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_rq2_manifest_fails_closed_on_incomplete_state(tmp_path: Path) -> None:
    plan = {
        "protocol": "trajuq_repeated_sampling_plan_v2",
        "status": "PASS",
        "experiment_git_commit": EXPECTED_COMMIT,
        "expected_cells": 12,
        "cells": [],
    }
    write_json(tmp_path / "plan.json", plan)
    write_json(tmp_path / "state.json", {"status": "IN_PROGRESS", "macro": "MISSING"})

    with pytest.raises(ValueError, match="12-cell"):
        build(tmp_path, Path("plan.json"), Path("state.json"), Path("out.json"))


def test_sha_fixture_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_bytes(b"TrajUQ")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "6db24e0a421d34039ea707f77b2d1006f119647ea2c01d5442a99bd02ec9ca6d"
