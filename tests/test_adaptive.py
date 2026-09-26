import numpy as np

from trajuq.adaptive import (
    bootstrap_two_budget_policy,
    evaluate_two_budget_policy,
    select_threshold_under_budget,
)
from trajuq.calibration import cross_fitted_error_probabilities


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
    assert result.mean_compute == 8.0


def test_threshold_selection_respects_budget():
    risk = np.array([0.1, 0.2, 0.8, 0.9])
    base = np.array([1, 1, 0, 0])
    refined = np.array([1, 1, 1, 1])
    result = select_threshold_under_budget(risk, base, refined, 0.5, 4, 8)
    assert result.refine_fraction <= 0.5
    assert result.accuracy == 1.0



def test_cross_fitted_probabilities_are_out_of_fold_and_finite():
    x = np.array(
        [[0.1], [0.2], [0.3], [0.4], [0.6], [0.7], [0.8], [0.9]],
        dtype=float,
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
    p, folds = cross_fitted_error_probabilities(x, y, n_splits=4, random_state=1)
    assert folds == 4
    assert p.shape == (8,)
    assert np.all(np.isfinite(p))
    assert np.all((p >= 0) & (p <= 1))



def test_adaptive_bootstrap_is_reproducible():
    risk = np.array([0.1, 0.2, 0.8, 0.9, 0.85, 0.15])
    base = np.array([1, 1, 0, 0, 0, 1])
    refined = np.array([1, 1, 1, 1, 1, 1])
    a = bootstrap_two_budget_policy(
        risk,
        base,
        refined,
        threshold=0.5,
        base_compute=4,
        refined_compute=8,
        n_boot=100,
        seed=7,
    )
    b = bootstrap_two_budget_policy(
        risk,
        base,
        refined,
        threshold=0.5,
        base_compute=4,
        refined_compute=8,
        n_boot=100,
        seed=7,
    )
    assert a == b
    assert a.delta_vs_base_point > 0
    assert a.n_boot == 100
