from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_release_manifest import sha256


ROOT = Path(__file__).resolve().parents[1]


def test_authorized_release_artifact_hashes_match() -> None:
    payload = json.loads((ROOT / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    assert payload["experiment_commit"] == "28002055c828c6910840de6f2105aa6367514144"
    assert payload["pending_science"]
    for item in payload["artifacts"]:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert sha256(path) == item["sha256"], item["path"]
        assert item["status"].startswith("AUTHORIZED")


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

