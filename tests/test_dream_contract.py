from pathlib import Path


def test_dream_telemetry_uses_pre_transfer_logits_hook():
    text = Path("src/trajuq/adapters/dream.py").read_text(encoding="utf-8")

    assert "def logits_hook(step, x, logits):" in text
    assert "acc.observe(step=step, logits=logits, state_ids=x)" in text
    assert "return logits" in text
    assert "generation_logits_hook_func=logits_hook" in text
    assert "generation_tokens_hook_func=hook" not in text
    assert '"telemetry_hook": "pre_transfer_generation_logits_hook"' in text


def test_dream_documentation_explains_hook_timing():
    text = Path("docs/DREAM_INSTRUMENTATION.md").read_text(encoding="utf-8")

    assert "pre-transfer" in text
    assert "generation_logits_hook_func" in text
