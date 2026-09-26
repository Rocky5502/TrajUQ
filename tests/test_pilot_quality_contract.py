from pathlib import Path


def test_pilot_quality_gate_requires_gpu_memory_and_single_revision():
    text = Path("scripts/pilot_quality_gate.py").read_text(encoding="utf-8")
    assert "missing/non-positive peak_gpu_memory_bytes" in text
    assert "exactly one immutable model revision" in text
    assert 'observed_adapters != {"llada"}' in text
    assert "active_unresolved_positions_v4" in text
