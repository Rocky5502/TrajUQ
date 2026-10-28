from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rq2_sample_efficiency_is_nested_and_exploratory() -> None:
    contract = json.loads(
        (ROOT / "configs/analysis/rq2_sample_efficiency.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["status"] == "PREDECLARED_SECONDARY_EXPLORATORY_ANALYSIS"
    assert contract["confirmatory"] is False
    assert contract["timing"] == {
        "declared_after_source_generation": True,
        "declared_before_sample_count_subset_calculation": True,
        "outcome_based_subset_selection": False,
    }
    assert contract["sample_counts"] == {
        "2": [2027, 2028],
        "3": [2027, 2028, 2029],
        "5": [2027, 2028, 2029, 2030, 2031],
    }
    assert contract["subset_rule"] == (
        "fixed_nested_seed_prefixes_declared_before_subset_calculation"
    )
    assert "new_generation" in contract["forbidden"]
    assert "replacement_of_authorized_k5_rq2_result" in contract["forbidden"]


def test_rq2_sample_efficiency_executor_has_hard_guards() -> None:
    script = (ROOT / "scripts/analyze_rq2_sample_efficiency.py").read_text(
        encoding="utf-8"
    )
    assert "Refusing to overwrite" in script
    assert "K=5 does not reproduce authorized metrics" in script
    assert '"new_model_generations": 0' in script
    assert "secondary_exploratory" in script
    assert '"cost_fraction_of_k5"' in script
    assert '"metric_delta_vs_k5"' in script
