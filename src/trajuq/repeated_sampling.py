from __future__ import annotations

import math
from collections import Counter

import numpy as np


def vote_uncertainty(predictions: list[str | None]) -> dict[str, float]:
    """Label-free uncertainty from repeated final-answer samples."""

    if len(predictions) < 2:
        raise ValueError("Repeated-sampling uncertainty requires at least two samples.")

    canonical = [p if p is not None else "<PARSE_FAILURE>" for p in predictions]
    counts = Counter(canonical)
    n = len(canonical)
    max_count = max(counts.values())
    disagreement = 1.0 - max_count / n

    probabilities = np.asarray(list(counts.values()), dtype=float) / n
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    max_entropy = math.log(n)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "sample_disagreement": float(disagreement),
        "sample_vote_entropy": float(normalized_entropy),
        "sample_unique_fraction": float(len(counts) / n),
    }



def equivalence_cluster_labels(
    values: list[str],
    equivalent,
) -> list[str]:
    """Deterministically cluster repeated outputs under a pairwise equivalence test.

    With the frozen five-sample baseline this requires at most ten pairwise
    comparisons per example. Connected components are used so the final vote
    labels are invariant to sample ordering.
    """

    n = len(values)
    if n < 2:
        raise ValueError("Need at least two values to cluster.")

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        left = find(i)
        right = find(j)
        if left == right:
            return
        # Stable root makes labels deterministic across runs.
        if left < right:
            parent[right] = left
        else:
            parent[left] = right

    for i in range(n):
        for j in range(i + 1, n):
            if equivalent(values[i], values[j]):
                union(i, j)

    roots = [find(i) for i in range(n)]
    root_to_label = {
        root: f"<EQ{rank}>"
        for rank, root in enumerate(sorted(set(roots)))
    }
    return [root_to_label[root] for root in roots]
