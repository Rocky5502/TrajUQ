from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cross_regime_synthesis_requires_exact_matching() -> None:
    contract = json.loads(
        (ROOT / "configs/analysis/cross_model_synthesis.json").read_text(
            encoding="utf-8"
        )
    )
    policy = contract["matched_cross_regime_policy"]
    assert policy["status"] == "FROZEN_READINESS_ONLY_UNTIL_RQ4_AND_RQ5_FINAL"
    assert set(policy["required_exact_matches"]) == {
        "dataset_identity",
        "externally_scored_answer_error_outcome",
        "calibration_split_identity_and_hash",
        "test_split_identity_and_hash",
        "metric_definition_and_direction",
        "cell_aggregation_rule",
    }
    assert "causal_effect_of_observability_level" in policy["forbidden"]
    assert "imputation_of_pending_or_unmatched_cells" in policy["forbidden"]


def test_flare_remains_fail_closed() -> None:
    model = (ROOT / "configs/models/flare.yaml").read_text(encoding="utf-8")
    readiness = (ROOT / "docs/FLARE_READINESS.md").read_text(encoding="utf-8")
    assert "revision: null" in model
    assert "SKIP / NO-GO" in readiness
    assert "No seed-matched native-versus-instrumented exact-token PASS" in readiness
    assert "Not defensibly estimable" in readiness
    assert "Do not substitute a" in readiness
    assert "different checkpoint" in readiness
