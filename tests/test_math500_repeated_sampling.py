import pytest

from scripts.evaluate_repeated_sampling import uncertainty_features_for_outputs


def test_math500_repeated_sampling_uses_symbolic_clusters():
    pytest.importorskip("math_verify")
    features = uncertainty_features_for_outputs(
        "math500",
        [
            r"FINAL: \frac{1}{2}",
            r"FINAL: 0.5",
            r"FINAL: \frac{2}{4}",
            r"FINAL: \frac{3}{4}",
            r"FINAL: \frac{1}{2}",
        ],
    )
    assert features["sample_disagreement"] == pytest.approx(0.2)
    assert features["sample_unique_fraction"] == pytest.approx(0.4)
