from pathlib import Path


def test_inventory_capture_never_serializes_or_prints_secret():
    text = Path("scripts/capture_siliconflow_inventory.py").read_text(encoding="utf-8")
    assert "SILICONFLOW_API_KEY" in text
    assert '"authorization_persisted": False' in text
    assert "print(token)" not in text
    assert '"Authorization": f"Bearer {token}"' in text
    assert "https://api.siliconflow.cn/v1/models?sub_type=chat" in text


def test_inventory_targets_only_three_confirmatory_models():
    text = Path("scripts/capture_siliconflow_inventory.py").read_text(encoding="utf-8")
    assert "Pro/deepseek-ai/DeepSeek-V3.2" in text
    assert "Pro/zai-org/GLM-5.1" in text
    assert "Pro/moonshotai/Kimi-K2.6" in text
    assert "DeepSeek-V4" not in text
    assert "GLM-5.2" not in text
    assert "Kimi-K2.7" not in text
