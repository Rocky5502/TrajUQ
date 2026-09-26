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
