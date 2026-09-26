from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .evaluation import selective_curve


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    ci_low: float
    ci_high: float
    n_boot: int
    valid_boot: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_boot": self.n_boot,
            "valid_boot": self.valid_boot,
        }


def _metric(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if name == "auroc":
        from sklearn.metrics import roc_auc_score

        return lambda y, s: float(roc_auc_score(y, s))
    if name == "auprc":
        from sklearn.metrics import average_precision_score

        return lambda y, s: float(average_precision_score(y, s))
    if name == "aurc":
        return lambda y, s: float(selective_curve(y, s).aurc)
    raise ValueError(f"unsupported metric {name!r}")


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    groups = [np.flatnonzero(y == label) for label in np.unique(y)]
    sampled = [rng.choice(g, size=len(g), replace=True) for g in groups if len(g)]
    if not sampled:
        raise ValueError("cannot bootstrap empty labels")
    idx = np.concatenate(sampled)
    rng.shuffle(idx)
    return idx


def paired_bootstrap_difference(
    y_error: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    metric: str,
    n_boot: int = 2000,
    seed: int = 2027,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Paired class-stratified bootstrap for metric(A) - metric(B)."""

    y = np.asarray(y_error, dtype=int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("y_error, score_a, and score_b must have equal length")
    if len(y) == 0:
        raise ValueError("empty inputs")
    if len(np.unique(y)) < 2 and metric in {"auroc", "auprc"}:
        raise ValueError(f"{metric} needs both error classes")

    fn = _metric(metric)
    point = fn(y, a) - fn(y, b)
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    for _ in range(int(n_boot)):
        idx = stratified_bootstrap_indices(y, rng)
        try:
            value = fn(y[idx], a[idx]) - fn(y[idx], b[idx])
        except ValueError:
            continue
        if np.isfinite(value):
            diffs.append(float(value))

    if not diffs:
        raise ValueError("no valid bootstrap replicates")
    low, high = np.quantile(np.asarray(diffs), [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(point, float(low), float(high), int(n_boot), len(diffs))



def paired_bootstrap_improvement(
    y_error: np.ndarray,
    candidate_score: np.ndarray,
    baseline_score: np.ndarray,
    metric: str,
    n_boot: int = 2000,
    seed: int = 2027,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Paired bootstrap with a common direction: positive means candidate is better.

    AUROC/AUPRC are higher-is-better, so improvement is candidate - baseline.
    AURC is lower-is-better, so improvement is baseline - candidate.
    """

    raw = paired_bootstrap_difference(
        y_error,
        candidate_score,
        baseline_score,
        metric=metric,
        n_boot=n_boot,
        seed=seed,
        alpha=alpha,
    )
    if metric in {"auroc", "auprc"}:
        return raw
    if metric == "aurc":
        return BootstrapResult(
            point=-raw.point,
            ci_low=-raw.ci_high,
            ci_high=-raw.ci_low,
            n_boot=raw.n_boot,
            valid_boot=raw.valid_boot,
        )
    raise ValueError(f"unsupported metric {metric!r}")
