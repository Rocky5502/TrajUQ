import numpy as np

from trajuq.evaluation import calibration_metrics, error_detection_metrics, selective_curve


def test_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = error_detection_metrics(y, s)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0


def test_selective_curve_shapes():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    curve = selective_curve(y, s)
    assert curve.coverage.shape == curve.risk.shape
    assert 0.0 <= curve.aurc <= 1.0


def test_calibration_metrics_finite():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.2, 0.7])
    metrics = calibration_metrics(y, p)
    assert all(np.isfinite(list(metrics.values())))
