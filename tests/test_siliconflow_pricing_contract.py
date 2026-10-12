from pathlib import Path


def test_pricing_extractor_is_restricted_to_confirmatory_models():
    text = Path("scripts/extract_siliconflow_pricing.py").read_text(encoding="utf-8")
    assert "Pro/deepseek-ai/DeepSeek-V3.2" in text
    assert "Pro/zai-org/GLM-5.1" in text
    assert "Pro/moonshotai/Kimi-K2.6" in text
    assert "captured_page_sha256" in text
    assert "DeepSeek-V4" not in text
