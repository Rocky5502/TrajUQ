from pathlib import Path

import yaml


def test_budget_addendum_changes_only_operational_cost_authority():
    contract = yaml.safe_load(
        Path("configs/operations/rq5_budget_authorization_af0347a.yaml").read_text(encoding="utf-8")
    )
    assert contract["freeze_sha256"] == "af0347a777d4a683c158407b9d0c53b7bff7236b8d35a63129077f6848886abc"
    assert contract["scope"] == "execution_cost_only_no_scientific_change"
    assert contract["preferred_target_rmb"] == 1200.0
    assert contract["warning_threshold_rmb"] == 1500.0
    assert contract["absolute_hard_stop_rmb"] == 1600.0
    assert "spending_for_significance" in contract["forbidden"]


def test_live_ledger_is_secret_free_and_freeze_bound():
    text = Path("scripts/rq5_cost_ledger.py").read_text(encoding="utf-8")
    assert "freeze_sha256" in text
    assert "SILICONFLOW_API_KEY" not in text
    assert "Authorization" not in text
