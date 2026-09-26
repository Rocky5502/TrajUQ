from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptiveCell:
    label: str
    adapter: str
    dataset: str
    base_correct: np.ndarray
    refined_correct: np.ndarray
    selected: np.ndarray
    base_compute: float
    refined_compute: float

    def __post_init__(self) -> None:
        base = np.asarray(self.base_correct, dtype=float)
        refined = np.asarray(self.refined_correct, dtype=float)
        selected = np.asarray(self.selected, dtype=bool)
        if not (len(base) == len(refined) == len(selected)):
            raise ValueError(f"{self.label}: arrays must have equal length")
        if len(base) == 0:
            raise ValueError(f"{self.label}: empty adaptive cell")
        if self.base_compute <= 0 or self.refined_compute <= 0:
            raise ValueError(f"{self.label}: compute budgets must be positive")


def cell_statistics(
    cell: AdaptiveCell,
    idx: np.ndarray | None = None,
) -> dict[str, float]:
    base = np.asarray(cell.base_correct, dtype=float)
    refined = np.asarray(cell.refined_correct, dtype=float)
    selected = np.asarray(cell.selected, dtype=bool)
    if idx is not None:
        base = base[idx]
        refined = refined[idx]
        selected = selected[idx]

    fraction = float(selected.mean())
    chosen = np.where(selected, refined, base)
    adaptive_accuracy = float(chosen.mean())
    base_accuracy = float(base.mean())
    refined_accuracy = float(refined.mean())
    random_expected = float(
        np.mean((1.0 - fraction) * base + fraction * refined)
    )
    nominal_compute = float(
        cell.base_compute + fraction * cell.refined_compute
    )
    return {
        "adaptive_accuracy": adaptive_accuracy,
        "base_accuracy": base_accuracy,
        "refined_accuracy": refined_accuracy,
        "random_expected_accuracy": random_expected,
        "delta_vs_base": adaptive_accuracy - base_accuracy,
        "delta_vs_refined": adaptive_accuracy - refined_accuracy,
        "delta_vs_random": adaptive_accuracy - random_expected,
        "selected_fraction": fraction,
        "nominal_compute": nominal_compute,
        "nominal_cost_ratio_to_base": nominal_compute / cell.base_compute,
        "nominal_cost_ratio_to_refined": nominal_compute / cell.refined_compute,
        "refined_cost_ratio_to_base": cell.refined_compute / cell.base_compute,
    }


def hierarchical_macro_adaptive(
    cells: list[AdaptiveCell],
    n_boot: int = 2000,
    seed: int = 2027,
    alpha: float = 0.05,
) -> dict:
    """Equal-cell-weight macro RQ3 summary with paired within-cell bootstrap."""

    if not cells:
        raise ValueError("at least one adaptive cell is required")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    point_by_cell = {cell.label: cell_statistics(cell) for cell in cells}
    fields = tuple(next(iter(point_by_cell.values())).keys())
    macro_point = {
        field: float(np.mean([stats[field] for stats in point_by_cell.values()]))
        for field in fields
    }

    rng = np.random.default_rng(seed)
    boot: dict[str, list[float]] = {field: [] for field in fields}
    for _ in range(int(n_boot)):
        replicate: dict[str, list[float]] = {field: [] for field in fields}
        for cell in cells:
            n = len(cell.base_correct)
            idx = rng.integers(0, n, size=n)
            stats = cell_statistics(cell, idx)
            for field in fields:
                replicate[field].append(stats[field])
        for field in fields:
            boot[field].append(float(np.mean(replicate[field])))

    interval_fields = (
        "adaptive_accuracy",
        "delta_vs_base",
        "delta_vs_refined",
        "delta_vs_random",
        "selected_fraction",
        "nominal_cost_ratio_to_base",
    )
    intervals: dict[str, dict[str, float]] = {}
    for field in interval_fields:
        values = np.asarray(boot[field], dtype=float)
        low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
        intervals[field] = {
            "point": macro_point[field],
            "ci_low": float(low),
            "ci_high": float(high),
        }

    family_delta: dict[str, list[float]] = {}
    for cell in cells:
        family_delta.setdefault(cell.adapter, []).append(
            point_by_cell[cell.label]["delta_vs_random"]
        )

    return {
        "macro": macro_point,
        "intervals": intervals,
        "n_cells": len(cells),
        "n_boot": int(n_boot),
        "model_family_delta_vs_random": {
            family: float(np.mean(values))
            for family, values in sorted(family_delta.items())
        },
        "per_cell": point_by_cell,
    }
