from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


HYBRID_TRACE_SCHEMA_VERSION = "trajuq-hybrid-native-trace-v1"


@dataclass(frozen=True)
class HybridTraceSummary:
    request_id: str
    features: dict[str, float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HYBRID_TRACE_SCHEMA_VERSION,
            "request_id": self.request_id,
            "features": self.features,
            "metadata": self.metadata,
        }


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def topk_entropy_proxy(top_logits: Iterable[float]) -> float | None:
    """Normalized entropy inside the traced top-k support.

    HybridDiffusion's native trace exports only top-k logits, not the full
    vocabulary distribution. This quantity is therefore explicitly a proxy and
    must never be reported as full predictive entropy.
    """

    values = np.asarray(list(top_logits), dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        return None
    shifted = values - float(values.max())
    weights = np.exp(shifted)
    total = float(weights.sum())
    if total <= 0 or not math.isfinite(total):
        return None
    probs = weights / total
    entropy = -float(np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    return float(entropy / math.log(len(values)))


def group_hybrid_trace(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        if "req_pool_idx" not in record:
            raise ValueError(f"trace row {index} is missing req_pool_idx")
        grouped[str(record["req_pool_idx"])].append(record)
    return dict(grouped)


def summarize_hybrid_request(
    request_id: str,
    records: list[dict[str, Any]],
    late_fraction: float = 0.25,
) -> HybridTraceSummary:
    if not records:
        raise ValueError("cannot summarize an empty HybridDiffusion trace")
    if not 0.0 < late_fraction <= 1.0:
        raise ValueError("late_fraction must be in (0, 1]")

    def order_key(row: dict[str, Any]) -> tuple[int, int, int]:
        return (
            int(row.get("forward_id", -1)),
            int(row.get("batch_idx", -1)),
            int(row.get("output_pos", -1)),
        )

    rows = sorted(records, key=order_key)
    forward_ids = sorted({int(row.get("forward_id", -1)) for row in rows})
    late_count = max(1, int(math.ceil(len(forward_ids) * late_fraction)))
    late_forwards = set(forward_ids[-late_count:])
    late_rows = [row for row in rows if int(row.get("forward_id", -1)) in late_forwards]

    sources = [str(row.get("source", "unknown")) for row in rows]
    accepted_spec = sum(source == "accepted_spec" for source in sources)
    corrections = sum(source == "correction" for source in sources)
    clean_samples = sum(source == "clean_sample" for source in sources)
    t0_rows = sum(source == "t0" for source in sources)
    speculative_decisions = accepted_spec + corrections

    corrected_forwards = {
        int(row.get("forward_id", -1))
        for row in rows
        if str(row.get("source")) == "correction"
    }
    accepted_by_forward = [
        sum(
            str(row.get("source")) == "accepted_spec"
            for row in rows
            if int(row.get("forward_id", -1)) == forward_id
        )
        for forward_id in forward_ids
    ]

    margins = [
        float(row["argmax_margin"])
        for row in rows
        if row.get("argmax_margin") is not None
        and math.isfinite(float(row["argmax_margin"]))
    ]
    late_margins = [
        float(row["argmax_margin"])
        for row in late_rows
        if row.get("argmax_margin") is not None
        and math.isfinite(float(row["argmax_margin"]))
    ]

    entropy_proxy: list[float] = []
    late_entropy_proxy: list[float] = []
    topk_sizes: list[int] = []
    for row in rows:
        logits = row.get("top_logits")
        if not isinstance(logits, list):
            continue
        topk_sizes.append(len(logits))
        value = topk_entropy_proxy(logits)
        if value is None:
            continue
        entropy_proxy.append(value)
        if int(row.get("forward_id", -1)) in late_forwards:
            late_entropy_proxy.append(value)

    accepted_flags = [row.get("accepted") for row in rows if row.get("accepted") is not None]
    explicit_accepts = sum(bool(value) for value in accepted_flags)
    explicit_rejects = sum(not bool(value) for value in accepted_flags)

    features = {
        "trace_rows": float(len(rows)),
        "trace_forwards": float(len(forward_ids)),
        "accepted_spec_fraction": _safe_fraction(accepted_spec, speculative_decisions),
        "correction_fraction": _safe_fraction(corrections, len(rows)),
        "reject_forward_fraction": _safe_fraction(len(corrected_forwards), len(forward_ids)),
        "clean_sample_fraction": _safe_fraction(clean_samples, len(rows)),
        "t0_fraction": _safe_fraction(t0_rows, len(rows)),
        "accepted_specs_per_forward_mean": float(np.mean(accepted_by_forward)),
        "argmax_margin_mean": float(np.mean(margins)) if margins else 0.0,
        "argmax_margin_p10": float(np.quantile(margins, 0.10)) if margins else 0.0,
        "argmax_margin_min": float(np.min(margins)) if margins else 0.0,
        "late_argmax_margin_mean": float(np.mean(late_margins)) if late_margins else 0.0,
        "topk_entropy_proxy_mean": float(np.mean(entropy_proxy)) if entropy_proxy else 0.0,
        "topk_entropy_proxy_p90": float(np.quantile(entropy_proxy, 0.90)) if entropy_proxy else 0.0,
        "late_topk_entropy_proxy_mean": (
            float(np.mean(late_entropy_proxy)) if late_entropy_proxy else 0.0
        ),
        "explicit_accept_fraction": _safe_fraction(
            explicit_accepts,
            explicit_accepts + explicit_rejects,
        ),
        "late_correction_fraction": _safe_fraction(
            sum(str(row.get("source")) == "correction" for row in late_rows),
            len(late_rows),
        ),
    }

    if not all(math.isfinite(float(value)) for value in features.values()):
        raise ValueError("non-finite HybridDiffusion diagnostic feature")

    metadata = {
        "source": "HybridDiffusion native SGLang trace",
        "late_fraction": float(late_fraction),
        "speculative_decision_rows": int(speculative_decisions),
        "accepted_spec_rows": int(accepted_spec),
        "correction_rows": int(corrections),
        "explicit_accept_rows": int(explicit_accepts),
        "explicit_reject_rows": int(explicit_rejects),
        "rows_with_argmax_margin": int(len(margins)),
        "rows_with_topk_entropy_proxy": int(len(entropy_proxy)),
        "trace_topk_min": int(min(topk_sizes)) if topk_sizes else 0,
        "trace_topk_max": int(max(topk_sizes)) if topk_sizes else 0,
        "entropy_semantics": (
            "topk_entropy_proxy is normalized entropy after renormalizing only the "
            "native trace's top-k logits; it is NOT full-vocabulary predictive entropy"
        ),
    }
    return HybridTraceSummary(str(request_id), features, metadata)


def summarize_hybrid_trace(
    records: Iterable[dict[str, Any]],
    late_fraction: float = 0.25,
) -> list[HybridTraceSummary]:
    grouped = group_hybrid_trace(records)
    return [
        summarize_hybrid_request(request_id, rows, late_fraction=late_fraction)
        for request_id, rows in sorted(grouped.items())
    ]
