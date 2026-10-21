from __future__ import annotations

import json
from pathlib import Path

import pytest

from trajuq.cross_model_synthesis import (
    AVAILABLE,
    NOT_APPLICABLE,
    PENDING,
    assert_compatible_for_direct_comparison,
    build_registry,
    evidence_value,
)


ROOT = Path(__file__).resolve().parents[1]


def test_evidence_states_do_not_conflate_pending_na_and_zero():
    measured_zero = evidence_value(AVAILABLE, 0.0)
    pending = evidence_value(PENDING, reason="not complete")
    unavailable = evidence_value(NOT_APPLICABLE, reason="not exposed")

    assert measured_zero["value"] == 0.0
    assert pending["value"] is None
    assert unavailable["value"] is None
    assert len({measured_zero["state"], pending["state"], unavailable["state"]}) == 3


def test_available_evidence_requires_a_value():
    with pytest.raises(ValueError, match="requires a value"):
        evidence_value(AVAILABLE)


def test_metric_compatibility_firewall_rejects_cross_regime_ranking():
    diffusion = {"compatibility_group": "open_diffusion_primary_v1"}
    black_box = {"compatibility_group": "hosted_black_box_rq5_v1"}
    with pytest.raises(ValueError, match="firewall"):
        assert_compatible_for_direct_comparison(diffusion, black_box)


def test_metric_compatibility_firewall_allows_same_registered_group():
    left = {"compatibility_group": "open_diffusion_primary_v1"}
    right = {"compatibility_group": "open_diffusion_primary_v1"}
    assert_compatible_for_direct_comparison(left, right)


def test_current_registry_is_seven_models_and_provisional():
    contract = ROOT / "configs" / "analysis" / "cross_model_synthesis.json"
    registry = build_registry(ROOT, contract)
    assert registry["schema"] == "trajuq_cross_model_synthesis_v1"
    assert registry["status"] == "PROVISIONAL"
    assert registry["final"] is False
    assert [model["key"] for model in registry["models"]] == [
        "dream",
        "llada",
        "sdar",
        "llama31",
        "deepseek_v32",
        "glm51",
        "kimi_k26",
    ]
    assert len(registry["appendix_rows"]) == 42
    for model in registry["models"][:3]:
        assert model["metrics"]["auroc"]["state"] == AVAILABLE
    for model in registry["models"][3:]:
        assert model["metrics"]["auroc"]["state"] == PENDING


def test_contract_explicitly_forbids_cross_regime_ranking():
    contract = json.loads(
        (ROOT / "configs" / "analysis" / "cross_model_synthesis.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["finalization_rule"]["forbid_cross_regime_statistical_ranking"]
    assert contract["finalization_rule"]["regenerate_from_scratch"]
