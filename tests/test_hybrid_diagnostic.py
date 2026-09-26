import numpy as np

from scripts.evaluate_hybrid_diagnostic import disagreement_groups


def test_hybrid_disagreement_group_error_rates():
    y = np.array([0, 1, 1, 0], dtype=int)
    x = np.zeros((4, 7), dtype=float)
    # parsed_answer_available
    x[:, 1] = 1.0
    # parsed_answer_disagreement on rows 1 and 2
    x[1:3, 2] = 1.0

    result = disagreement_groups(y, x)
    assert result["agreement_n"] == 2
    assert result["disagreement_n"] == 2
    assert result["agreement_error_rate"] == 0.0
    assert result["disagreement_error_rate"] == 1.0
    assert result["disagreement_minus_agreement_error_rate"] == 1.0
