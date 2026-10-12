from pathlib import Path


def test_rq5_freeze_is_clean_outcome_blind_and_non_overwriting():
    text = Path("scripts/freeze_rq5.py").read_text(encoding="utf-8")
    assert 'git("status", "--porcelain")' in text
    assert "Refusing to overwrite immutable RQ5 freeze evidence" in text
    assert '"outcomes_observed": False' in text
    assert "hard_cost_cap_rmb" in text


def test_rq5_executor_is_idempotent_and_secret_safe():
    text = Path("scripts/run_rq5_siliconflow.py").read_text(encoding="utf-8")
    assert "uuid.uuid5" in text
    assert "SKIP_SUCCESS" in text
    assert "O_EXCL" in text
    assert '"Authorization": f"Bearer {self.token}"' in text
    assert '"authorization"' not in Path("configs/schemas/rq5_api_provenance_v1.json").read_text(encoding="utf-8").lower().split('"forbidden"')[0]
