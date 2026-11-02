from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_release_manifest import sha256


ROOT = Path(__file__).resolve().parents[1]


def test_authorized_release_artifact_hashes_match() -> None:
    payload = json.loads((ROOT / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    assert payload["experiment_commit"] == "28002055c828c6910840de6f2105aa6367514144"
    assert payload["closure_classification"] == "FINAL_CLOSURE"
    assert payload["pending_science"] == []
    allowed_statuses = {
        "AUTHORIZED",
        "AUTHORIZED_FINAL",
        "AUTHORIZED_PRIMARY",
        "AUTHORIZED_SENSITIVITY",
        "AUTHORIZED_SNAPSHOT",
        "DESCRIPTIVE_ONLY",
        "DESCRIPTIVE_SYNTHESIS",
        "FROZEN",
        "PRESERVED",
        "RECOVERED_SEPARATE",
        "SECONDARY_EXPLORATORY",
        "SECONDARY_EXPLORATORY_SYSTEMS_DIAGNOSTIC",
    }
    for item in payload["artifacts"]:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert sha256(path) == item["sha256"], item["path"]
        assert item["status"] in allowed_statuses


def test_released_rq3_evidence_is_complete() -> None:
    manifest = json.loads((ROOT / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    assert "RQ3 adaptive aggregate" not in manifest["pending_science"]

    state = json.loads(
        (ROOT / "results/adaptive_execution_state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "COMPLETE"
    assert state["complete_cells"] == state["observed_cells"] == state["expected_cells"] == 18
    assert state["macros"] == {
        "cap_025": "COMPLETE",
        "cap_050": "COMPLETE",
        "cap_075": "COMPLETE",
    }

    plan = json.loads(
        (ROOT / "results/adaptive_matrix_plan.json").read_text(encoding="utf-8")
    )
    assert plan["status"] == "PASS"
    assert plan["cell_count"] == plan["expected_cells"] == 18

    for fraction in ("025", "050", "075"):
        aggregate = json.loads(
            (ROOT / f"results/adaptive/macro_cap_{fraction}.json").read_text(
                encoding="utf-8"
            )
        )
        assert aggregate["cell_set_complete"] is True
        assert aggregate["n_cells"] == aggregate["observed_cells"] == aggregate["expected_cells"] == 18


def test_released_rq2_evidence_is_complete() -> None:
    manifest = json.loads((ROOT / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    assert "RQ2 final aggregate" not in manifest["pending_science"]

    plan = json.loads((ROOT / "results/repeated_sampling_plan.json").read_text(encoding="utf-8"))
    assert plan["status"] == "PASS"
    assert plan["cell_count"] == plan["expected_cells"] == 12
    assert plan["seeds"] == [2027, 2028, 2029, 2030, 2031]

    state = json.loads(
        (ROOT / "results/repeated_sampling_execution_state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == state["macro"] == "COMPLETE"
    assert state["complete_cells"] == 12

    aggregate = json.loads(
        (ROOT / "results/repeated_sampling/macro_trajuq_vs_repeated.json").read_text(
            encoding="utf-8"
        )
    )
    assert aggregate["protocol"] == "trajuq_vs_repeated_macro_v2"
    assert aggregate["cell_set_complete"] is True
    assert aggregate["observed_cells"] == aggregate["expected_cells"] == 12

    final = json.loads(
        (ROOT / "results/generated/final/rq2_manifest.json").read_text(encoding="utf-8")
    )
    assert final["rq"] == "RQ2"
    assert final["status"] == "FINAL"
    assert final["audit_status"] == "PASS"
    assert final["gates"]["seed_jobs_complete"] == final["gates"]["seed_jobs_expected"] == 120
    assert final["gates"]["partitions_complete"] == final["gates"]["partitions_expected"] == 24

