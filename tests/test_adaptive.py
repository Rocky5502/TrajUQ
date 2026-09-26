import numpy as np

from trajuq.adaptive import evaluate_two_budget_policy, select_threshold_under_budget


def test_adaptive_policy_refines_high_risk_cases():
    risk = np.array([0.1, 0.2, 0.8, 0.9])
    base = np.array([1, 1, 0, 0])
    refined = np.array([1, 1, 1, 1])
    result = evaluate_two_budget_policy(
        risk,
        base,
        refined,
        threshold=0.5,
        base_compute=4,
        refined_compute=8,
    )
    assert result.accuracy == 1.0
    assert result.refine_fraction == 0.5
    assert result.mean_compute == 6.0


def test_threshold_selection_respects_budget():
    risk = np.array([0.1, 0.2, 0.8, 0.9])
    base = np.array([1, 1, 0, 0])
    refined = np.array([1, 1, 1, 1])
    result = select_threshold_under_budget(risk, base, refined, 0.5, 4, 8)
    assert result.refine_fraction <= 0.5
    assert result.accuracy == 1.0
