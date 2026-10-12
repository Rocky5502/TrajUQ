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


def test_budget_watch_has_absolute_stop_path():
    text = Path("scripts/watch_rq5_budget.py").read_text(encoding="utf-8")
    assert 'ledger["cost_watch_mode"] == "HARD_STOP"' in text
    assert 'not ledger["paid_execution_allowed"]' in text
    assert "signal.SIGTERM" in text


def test_executor_honors_addendum_and_never_resubmits_ambiguous_requests():
    text = Path("scripts/run_rq5_siliconflow.py").read_text(encoding="utf-8")
    assert "self.absolute_hard_cap" in text
    assert "AMBIGUOUS_MISSING_NOT_RESUBMITTED" in text
    assert '"will_resubmit": False' in text
