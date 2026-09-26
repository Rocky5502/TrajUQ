from pathlib import Path


def test_compare_calibrated_uses_frozen_feature_constants():
    text = Path("scripts/compare_calibrated.py").read_text(encoding="utf-8")
    assert "PRIMARY_TRAJECTORY_FEATURES" in text
    assert "FINAL_BASELINE_FEATURES" in text
    assert "TRAJECTORY_PLUS_FINAL_BASELINE_FEATURES" not in text
    assert "trajectory+decision logistic" in text
