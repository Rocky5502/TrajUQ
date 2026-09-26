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
    cost_mode: str = "selective_second_pass_rerun"

    def to_dict(self) -> dict[str, float | str]:
        return self.__dict__.copy()


def evaluate_two_budget_policy(
    risk: np.ndarray,
    base_correct: np.ndarray,
    refined_correct: np.ndarray,
    threshold: float,
    base_compute: float,
    refined_compute: float,
    cost_mode: str = "selective_second_pass_rerun",
) -> AdaptivePolicyResult:
    risk = np.asarray(risk, dtype=float)
    base = np.asarray(base_correct, dtype=float)
    refined = np.asarray(refined_correct, dtype=float)
    if not (len(risk) == len(base) == len(refined)):
        raise ValueError("risk and correctness arrays must have equal length")
    if len(risk) == 0:
        raise ValueError("empty inputs")
    if refined_compute <= 0 or base_compute <= 0:
        raise ValueError("compute values must be positive")
    if cost_mode not in {"selective_second_pass_rerun", "continuation"}:
        raise ValueError(f"unsupported cost_mode={cost_mode!r}")
    if cost_mode == "continuation" and refined_compute < base_compute:
        raise ValueError("continuation requires refined_compute >= base_compute")

    use_refined = risk > float(threshold)
    chosen = np.where(use_refined, refined, base)
    fraction = float(use_refined.mean())
    if cost_mode == "selective_second_pass_rerun":
        # The measured base output already exists before risk is known. If risk
        # triggers, a second high-budget generation is launched from scratch.
        # This is directly realizable with the paired frozen artifacts.
        mean_compute = float(base_compute + fraction * refined_compute)
    else:
        # Reserved for a runtime that truly continues the same trajectory from
        # the base state. Independent base/refined runs must not use this mode.
        mean_compute = float(
            base_compute + fraction * (refined_compute - base_compute)
        )
    random_expected = float(np.mean((1.0 - fraction) * base + fraction * refined))
    accuracy = float(chosen.mean())
    return AdaptivePolicyResult(
        threshold=float(threshold),
        refine_fraction=fraction,
        mean_compute=mean_compute,
        accuracy=accuracy,
        random_refine_expected_accuracy=random_expected,
        delta_vs_random=accuracy - random_expected,
        cost_mode=cost_mode,
    )


def select_threshold_under_budget(
    risk: np.ndarray,
    base_correct: np.ndarray,
    refined_correct: np.ndarray,
    max_refine_fraction: float,
    base_compute: float,
    refined_compute: float,
    cost_mode: str = "selective_second_pass_rerun",
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
            cost_mode=cost_mode,
        )
        if result.refine_fraction <= max_refine_fraction + 1e-12:
            feasible.append(result)

    if not feasible:
        raise ValueError("no feasible threshold")
    return max(feasible, key=lambda r: (r.accuracy, -r.mean_compute, r.threshold))



@dataclass(frozen=True)
class AdaptiveBootstrapResult:
    delta_vs_random_point: float
    delta_vs_random_ci_low: float
    delta_vs_random_ci_high: float
    delta_vs_base_point: float
    delta_vs_base_ci_low: float
    delta_vs_base_ci_high: float
    delta_vs_refined_point: float
    delta_vs_refined_ci_low: float
    delta_vs_refined_ci_high: float
    n_boot: int

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def bootstrap_two_budget_policy(
    risk: np.ndarray,
    base_correct: np.ndarray,
    refined_correct: np.ndarray,
    threshold: float,
    base_compute: float,
    refined_compute: float,
    n_boot: int = 2000,
    seed: int = 2027,
    alpha: float = 0.05,
    cost_mode: str = "selective_second_pass_rerun",
) -> AdaptiveBootstrapResult:
    """Paired bootstrap for a frozen two-budget policy on held-out examples."""

    risk = np.asarray(risk, dtype=float)
    base = np.asarray(base_correct, dtype=float)
    refined = np.asarray(refined_correct, dtype=float)
    if not (len(risk) == len(base) == len(refined)):
        raise ValueError("risk and correctness arrays must have equal length")
    if len(risk) == 0:
        raise ValueError("empty inputs")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    point = evaluate_two_budget_policy(
        risk,
        base,
        refined,
        threshold=threshold,
        base_compute=base_compute,
        refined_compute=refined_compute,
        cost_mode=cost_mode,
    )
    rng = np.random.default_rng(seed)
    vs_random: list[float] = []
    vs_base: list[float] = []
    vs_refined: list[float] = []

    n = len(risk)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boot = evaluate_two_budget_policy(
            risk[idx],
            base[idx],
            refined[idx],
            threshold=threshold,
            base_compute=base_compute,
            refined_compute=refined_compute,
            cost_mode=cost_mode,
        )
        vs_random.append(float(boot.delta_vs_random))
        vs_base.append(float(boot.accuracy - base[idx].mean()))
        vs_refined.append(float(boot.accuracy - refined[idx].mean()))

    q_low, q_high = alpha / 2, 1 - alpha / 2
    random_ci = np.quantile(np.asarray(vs_random), [q_low, q_high])
    base_ci = np.quantile(np.asarray(vs_base), [q_low, q_high])
    refined_ci = np.quantile(np.asarray(vs_refined), [q_low, q_high])

    return AdaptiveBootstrapResult(
        delta_vs_random_point=float(point.delta_vs_random),
        delta_vs_random_ci_low=float(random_ci[0]),
        delta_vs_random_ci_high=float(random_ci[1]),
        delta_vs_base_point=float(point.accuracy - base.mean()),
        delta_vs_base_ci_low=float(base_ci[0]),
        delta_vs_base_ci_high=float(base_ci[1]),
        delta_vs_refined_point=float(point.accuracy - refined.mean()),
        delta_vs_refined_ci_low=float(refined_ci[0]),
        delta_vs_refined_ci_high=float(refined_ci[1]),
        n_boot=int(n_boot),
    )
