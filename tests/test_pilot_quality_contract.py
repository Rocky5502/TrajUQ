from pathlib import Path


def test_pilot_quality_gate_requires_gpu_memory_and_single_revision():
    text = Path("scripts/pilot_quality_gate.py").read_text(encoding="utf-8")
    assert "missing/non-positive peak_gpu_memory_bytes" in text
    assert "exactly one immutable model revision" in text
    assert 'observed_adapters != {"llada"}' in text
    assert "active_unresolved_positions_v4" in text


def test_pilot_quality_gate_records_clean_checkout():
    text = Path("scripts/pilot_quality_gate.py").read_text(encoding="utf-8")
    assert "def git_clean()" in text
    assert "git worktree must be clean for the frozen pilot gate" in text
    assert '"git_clean": clean' in text
