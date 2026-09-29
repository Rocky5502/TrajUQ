from pathlib import Path


def test_sdar_native_parity_contract_uses_sampling_temperature():
    reference = Path("scripts/run_sdar_reference.py").read_text(encoding="utf-8")
    config = Path("configs/models/sdar.yaml").read_text(encoding="utf-8")
    docs = Path("docs/SDAR_INSTRUMENTATION.md").read_text(encoding="utf-8")

    assert '--temperature", type=float, default=1.0' in reference
    assert "torch.multinomial" in reference
    assert "temperature: 1.0" in config
    assert "run_sdar_reference.py" in docs
    assert "exact token" in docs.lower()


def test_sdar_parity_runs_and_checks_self_repeatability():
    wrapper = Path("scripts/parity_remaining_linux.sh").read_text(encoding="utf-8")
    assert "sdar_parity_reference_repeat.jsonl" in wrapper
    assert "sdar_parity_trajuq_repeat.raw.jsonl" in wrapper
    assert "Exact SDAR native self-repeatability" in wrapper
    assert "Exact SDAR instrumented self-repeatability" in wrapper
