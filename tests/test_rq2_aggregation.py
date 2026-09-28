import numpy as np

from trajuq.aggregation import metric_value, oriented_improvement


def test_calibration_metrics_are_lower_is_better():
    # Use enough examples that 80% coverage is strictly below full coverage.
    # With n=4, ceil(0.8 * n) == n, making risk80 necessarily equal to the
    # full-set error rate regardless of ranking and therefore unable to test
    # score orientation.
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=int)
    good = np.array(
        [0.02, 0.05, 0.10, 0.15, 0.20, 0.80, 0.85, 0.90, 0.95, 0.98],
        dtype=float,
    )
    bad = 1.0 - good

    for metric in ("brier", "nll", "ece15", "risk50", "risk80", "aurc"):
        assert oriented_improvement(y, good, bad, metric) > 0


def test_metric_value_supports_rq2_outputs():
    y = np.array([0, 0, 1, 1], dtype=int)
    p = np.array([0.1, 0.2, 0.8, 0.9], dtype=float)
    for metric in ("brier", "nll", "ece15", "risk50", "risk80"):
        value = metric_value(y, p, metric)
        assert np.isfinite(value)
