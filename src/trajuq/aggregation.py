from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluation import selective_curve
from .statistics import stratified_bootstrap_indices


@dataclass(frozen=True)
class PredictionCell:
    label: str
    y_error: np.ndarray
    candidate_score: np.ndarray
    baseline_score: np.ndarray
    model_family: str | None = None
    dataset: str | None = None

    def __post_init__(self):
        y = np.asarray(self.y_error, dtype=int)
        candidate = np.asarray(self.candidate_score, dtype=float)
        baseline = np.asarray(self.baseline_score, dtype=float)
        if not (len(y) == len(candidate) == len(baseline)):
            raise ValueError(f"{self.label}: prediction arrays must have equal length")
        if len(y) == 0:
            raise ValueError(f"{self.label}: empty prediction cell")
        if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(baseline)):
            raise ValueError(f"{self.label}: non-finite prediction score")


def metric_value(y_error: np.ndarray, score: np.ndarray, metric: str) -> float:
    y = np.asarray(y_error, dtype=int)
    s = np.asarray(score, dtype=float)
    if metric == "auroc":
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y)) < 2:
            raise ValueError("AUROC requires both error classes")
        return float(roc_auc_score(y, s))
    if metric == "auprc":
        from sklearn.metrics import average_precision_score

        if len(np.unique(y)) < 2:
            raise ValueError("AUPRC requires both error classes")
        return float(average_precision_score(y, s))
    if metric == "aurc":
        return float(selective_curve(y, s).aurc)
    raise ValueError(f"unsupported metric {metric!r}")


def oriented_improvement(
    y_error: np.ndarray,
    candidate_score: np.ndarray,
    baseline_score: np.ndarray,
    metric: str,
) -> float:
    candidate = metric_value(y_error, candidate_score, metric)
    baseline = metric_value(y_error, baseline_score, metric)
    if metric in {"auroc", "auprc"}:
        return float(candidate - baseline)
    if metric == "aurc":
        return float(baseline - candidate)
    raise ValueError(f"unsupported metric {metric!r}")


def hierarchical_macro_bootstrap(
    cells: list[PredictionCell],
    metric: str,
    n_boot: int = 2000,
    seed: int = 2027,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Bootstrap examples within each cell, then macro-average cells.

    Each model-dataset cell receives equal macro weight regardless of its number
    of benchmark examples. The same resampled indices are used for candidate
    and baseline within a cell, preserving the paired comparison.
    """

    if not cells:
        raise ValueError("at least one prediction cell is required")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    point_deltas = [
        oriented_improvement(
            cell.y_error,
            cell.candidate_score,
            cell.baseline_score,
            metric,
        )
        for cell in cells
    ]
    candidate_points = [
        metric_value(cell.y_error, cell.candidate_score, metric) for cell in cells
    ]
    baseline_points = [
        metric_value(cell.y_error, cell.baseline_score, metric) for cell in cells
    ]

    rng = np.random.default_rng(seed)
    boot_macro: list[float] = []
    for _ in range(int(n_boot)):
        cell_deltas: list[float] = []
        for cell in cells:
            y = np.asarray(cell.y_error, dtype=int)
            idx = stratified_bootstrap_indices(y, rng)
            cell_deltas.append(
                oriented_improvement(
                    y[idx],
                    np.asarray(cell.candidate_score, dtype=float)[idx],
                    np.asarray(cell.baseline_score, dtype=float)[idx],
                    metric,
                )
            )
        boot_macro.append(float(np.mean(cell_deltas)))

    low, high = np.quantile(
        np.asarray(boot_macro, dtype=float),
        [alpha / 2, 1 - alpha / 2],
    )
    positive_cells = int(sum(delta > 0 for delta in point_deltas))
    zero_cells = int(sum(np.isclose(delta, 0.0) for delta in point_deltas))

    family_deltas: dict[str, list[float]] = {}
    for cell, delta in zip(cells, point_deltas):
        if cell.model_family:
            family_deltas.setdefault(cell.model_family, []).append(float(delta))
    family_macro = {
        family: float(np.mean(values))
        for family, values in sorted(family_deltas.items())
    }
    positive_families = int(sum(value > 0 for value in family_macro.values()))

    return {
        "candidate_macro": float(np.mean(candidate_points)),
        "baseline_macro": float(np.mean(baseline_points)),
        "improvement_macro": float(np.mean(point_deltas)),
        "improvement_ci_low": float(low),
        "improvement_ci_high": float(high),
        "positive_cells": positive_cells,
        "zero_cells": zero_cells,
        "negative_cells": int(len(cells) - positive_cells - zero_cells),
        "n_cells": len(cells),
        "model_family_macro_improvement": family_macro,
        "positive_model_families": positive_families,
        "n_model_families": len(family_macro),
        "n_boot": int(n_boot),
    }
