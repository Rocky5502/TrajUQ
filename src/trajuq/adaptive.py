from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptivePolicyResult:
    threshold: float
    refine_fraction: float
    mean_compute: float
    accuracy: float
    random_refine_expected_accuracy: float
    delta_vs_random: float

    def to_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def evaluate_two_budget_policy(
    risk: np.ndarray,
    base_correct: np.ndarray,
    refined_correct: np.ndarray,
    threshold: float,
    base_compute: float,
    refined_compute: float,
) -> AdaptivePolicyResult:
    risk = np.asarray(risk, dtype=float)
    base = np.asarray(base_correct, dtype=float)
    refined = np.asarray(refined_correct, dtype=float)
    if not (len(risk) == len(base) == len(refined)):
        raise ValueError("risk and correctness arrays must have equal length")
    if len(risk) == 0:
        raise ValueError("empty inputs")
    if refined_compute < base_compute:
        raise ValueError("refined_compute must be >= base_compute")

    use_refined = risk > float(threshold)
    chosen = np.where(use_refined, refined, base)
    fraction = float(use_refined.mean())
    mean_compute = float(base_compute + fraction * (refined_compute - base_compute))
    random_expected = float(np.mean((1.0 - fraction) * base + fraction * refined))
    accuracy = float(chosen.mean())
    return AdaptivePolicyResult(
        threshold=float(threshold),
        refine_fraction=fraction,
        mean_compute=mean_compute,
        accuracy=accuracy,
        random_refine_expected_accuracy=random_expected,
        delta_vs_random=accuracy - random_expected,
    )


def select_threshold_under_budget(
    risk: np.ndarray,
    base_correct: np.ndarray,
    refined_correct: np.ndarray,
    max_refine_fraction: float,
    base_compute: float,
    refined_compute: float,
) -> AdaptivePolicyResult:
    """Select a threshold using calibration data only under a refinement cap."""

    if not 0.0 <= max_refine_fraction <= 1.0:
        raise ValueError("max_refine_fraction must be in [0,1]")
    risk = np.asarray(risk, dtype=float)
    candidates = np.unique(np.r_[np.inf, risk, -np.inf])
    feasible: list[AdaptivePolicyResult] = []
    for threshold in candidates:
        result = evaluate_two_budget_policy(
            risk,
            base_correct,
            refined_correct,
            threshold=float(threshold),
            base_compute=base_compute,
            refined_compute=refined_compute,
        )
        if result.refine_fraction <= max_refine_fraction + 1e-12:
            feasible.append(result)

    if not feasible:
        raise ValueError("no feasible threshold")
    return max(feasible, key=lambda r: (r.accuracy, -r.mean_compute, r.threshold))
