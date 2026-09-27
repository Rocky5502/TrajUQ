import numpy as np
import pytest

from trajuq.calibration import fit_isotonic_logistic_sensitivity


def test_isotonic_logistic_sensitivity_is_bounded_and_deterministic():
    x = np.column_stack(
        [
            np.linspace(-2.0, 2.0, 40),
            np.sin(np.linspace(0.0, 4.0, 40)),
        ]
    )
    y = np.asarray([0, 1] * 20, dtype=int)

    first, folds_first = fit_isotonic_logistic_sensitivity(
        x,
        y,
        n_splits=5,
        random_state=2027,
    )
    second, folds_second = fit_isotonic_logistic_sensitivity(
        x,
        y,
        n_splits=5,
        random_state=2027,
    )

    p_first = first.predict_proba(x)[:, 1]
    p_second = second.predict_proba(x)[:, 1]

    assert folds_first == 5
    assert folds_second == 5
    assert np.all(np.isfinite(p_first))
    assert np.all((0.0 <= p_first) & (p_first <= 1.0))
    assert np.allclose(p_first, p_second)


def test_isotonic_logistic_sensitivity_rejects_one_class():
    x = np.arange(12, dtype=float)[:, None]
    y = np.zeros(12, dtype=int)

    with pytest.raises(ValueError, match="both error classes"):
        fit_isotonic_logistic_sensitivity(x, y)


def test_isotonic_logistic_sensitivity_reduces_folds_to_minority_count():
    x = np.arange(12, dtype=float)[:, None]
    y = np.asarray([0] * 9 + [1] * 3, dtype=int)

    _, folds = fit_isotonic_logistic_sensitivity(
        x,
        y,
        n_splits=5,
        random_state=2027,
    )

    assert folds == 3
