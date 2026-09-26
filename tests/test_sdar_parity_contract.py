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
