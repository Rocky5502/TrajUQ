from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


@dataclass
class SelectiveCurve:
    coverage: np.ndarray
    risk: np.ndarray
    aurc: float


def expected_calibration_error(
    y_error: np.ndarray,
    p_error: np.ndarray,
    n_bins: int = 15,
) -> float:
    y = np.asarray(y_error, dtype=float)
    p = np.asarray(p_error, dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_error and p_error must have the same shape")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        mask = (p >= left) & (p < right if i < n_bins - 1 else p <= right)
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(ece)


def error_detection_metrics(y_error: np.ndarray, risk_score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_error, dtype=int)
    s = np.asarray(risk_score, dtype=float)
    if y.shape != s.shape:
        raise ValueError("y_error and risk_score must have the same shape")
    if len(np.unique(y)) < 2:
        return {"auroc": math.nan, "auprc": math.nan}
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
    }


def calibration_metrics(
    y_error: np.ndarray,
    p_error: np.ndarray,
    n_bins: int = 15,
) -> dict[str, float]:
    y = np.asarray(y_error, dtype=int)
    p = np.clip(np.asarray(p_error, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "brier": float(brier_score_loss(y, p)),
        "nll": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "ece": expected_calibration_error(y, p, n_bins=n_bins),
    }


def selective_curve(y_error: np.ndarray, risk_score: np.ndarray) -> SelectiveCurve:
    y = np.asarray(y_error, dtype=float)
    s = np.asarray(risk_score, dtype=float)
    if len(y) == 0:
        raise ValueError("empty arrays")
    order = np.argsort(s, kind="stable")
    accepted_errors = y[order]
    cumulative_errors = np.cumsum(accepted_errors)
    counts = np.arange(1, len(y) + 1)
    coverage = counts / len(y)
    risk = cumulative_errors / counts
    aurc = float(np.trapz(risk, coverage))
    return SelectiveCurve(coverage=coverage, risk=risk, aurc=aurc)


def risk_at_coverage(y_error: np.ndarray, risk_score: np.ndarray, coverage: float) -> float:
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    curve = selective_curve(y_error, risk_score)
    idx = min(
        len(curve.coverage) - 1,
        max(0, int(math.ceil(coverage * len(curve.coverage))) - 1),
    )
    return float(curve.risk[idx])
