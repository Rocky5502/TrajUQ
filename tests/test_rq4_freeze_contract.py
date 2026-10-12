from pathlib import Path


def test_rq4_freeze_is_clean_outcome_blind_and_non_overwriting():
    text = Path("scripts/freeze_rq4.py").read_text(encoding="utf-8")
    assert 'git("status", "--porcelain")' in text
    assert "Refusing to overwrite immutable RQ4 freeze evidence" in text
    assert '"outcomes_observed": False' in text
    assert "local_files_only=True" in text
    assert "rendered_prompt_sha256" in text
    assert 'parser.add_argument("--data-root"' in text


def test_rq4_configs_bind_the_freeze_producer_and_schema():
    experiment = Path("configs/experiments/tmlr_rq4.yaml").read_text(encoding="utf-8")
    model = Path("configs/models/llama31_ar.yaml").read_text(encoding="utf-8")
    assert "producer: scripts/freeze_rq4.py" in experiment
    assert "results/rq4/freeze_manifest.json" in experiment
    assert "configs/schemas/rq4_ar_telemetry_v1.json" in model
    assert "PENDING_PRE_OUTCOME_FREEZE" not in model
