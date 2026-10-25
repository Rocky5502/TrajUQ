import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_rq2_final_evidence import (
    EXPECTED_COMMIT,
    build,
    validate_finalization_counts,
)


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


def complete_count_fixture() -> tuple[dict, dict]:
    plan_cells = []
    state_cells = []
    for index in range(12):
        label = f"cell-{index:02d}"
        seeds = [2027, 2028, 2029, 2030, 2031]
        plan_cells.append({"cell": label, "seeds": seeds})

        def generation() -> dict:
            return {
                "complete": True,
                "manifest_valid": True,
                "seeds": {
                    str(seed): {
                        "complete": True,
                        "ids_complete": True,
                        "audit_pass": True,
                    }
                    for seed in seeds
                },
            }

        state_cells.append(
            {
                "cell": label,
                "calibration_generation": generation(),
                "test_generation": generation(),
            }
        )
    return {"cells": plan_cells}, {"cells": state_cells}


def test_final_gate_counts_recount_all_seed_and_partition_audits() -> None:
    plan, state = complete_count_fixture()
    assert validate_finalization_counts(plan, state) == {
        "seed_jobs_expected": 120,
        "seed_jobs_complete": 120,
        "partitions_expected": 24,
        "partitions_complete": 24,
    }


def test_final_gate_counts_fail_closed_on_one_seed_audit() -> None:
    plan, state = complete_count_fixture()
    state["cells"][7]["test_generation"]["seeds"]["2030"]["audit_pass"] = False
    with pytest.raises(ValueError, match="119/120"):
        validate_finalization_counts(plan, state)
