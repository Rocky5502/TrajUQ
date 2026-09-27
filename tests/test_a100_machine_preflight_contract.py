from pathlib import Path


def test_a100_machine_preflight_contract_is_fail_closed():
    text = Path("scripts/verify_a100_machine.py").read_text(encoding="utf-8")

    assert "trajuq_a100_machine_preflight_v1" in text
    assert 'default="A100"' in text
    assert "default=75.0" in text
    assert "default=40.0" in text
    assert "default=80.0" in text
    assert "torch.cuda.is_available()" in text
    assert "torch.cuda.is_bf16_supported()" in text
    assert "nvidia-smi" in text
    assert "git worktree is not clean" in text
    assert 'if payload["status"] != "PASS"' in text


def test_linux_pilot_runs_machine_preflight_before_dataset_preparation():
    text = Path("scripts/pilot_linux.sh").read_text(encoding="utf-8")

    machine = text.index("scripts/verify_a100_machine.py")
    prepare = text.index("scripts/prepare_eval_pair.py")
    assert machine < prepare


def test_a100_campaign_requires_machine_pass_for_readiness():
    text = Path("scripts/a100_campaign_linux.sh").read_text(encoding="utf-8")

    assert "require_a100_machine()" in text
    assert "results/a100_machine_preflight.json" in text
    assert "trajuq_a100_machine_preflight_v1" in text
    readiness = text.index("readiness)")
    assert "require_a100_machine" in text[readiness : readiness + 300]
