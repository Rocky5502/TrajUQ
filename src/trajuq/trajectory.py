from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .schema import StepSummary

try:
    import torch
except ImportError:
    torch = None


@dataclass
class _ObservedStep:
    step: int
    entropy: np.ndarray
    confidence: np.ndarray
    margin: np.ndarray
    js_drift: np.ndarray
    masked: np.ndarray
    active: np.ndarray
    prediction: np.ndarray
    state: np.ndarray


@dataclass
class TrajectoryFeaturePayload:
    """Mergeable sufficient statistics for an ordered denoising trajectory."""

    entropy_samples: np.ndarray
    confidence_samples: np.ndarray
    margin_samples: np.ndarray
    js_samples: np.ndarray

    decision_entropy: np.ndarray
    decision_confidence: np.ndarray
    decision_margin: np.ndarray
    commitment_time: np.ndarray

    step_entropy: np.ndarray
    step_confidence: np.ndarray
    step_margin: np.ndarray
    step_js: np.ndarray
    step_revision_rate: np.ndarray

    step_js_sum: np.ndarray
    step_js_count: np.ndarray
    step_revision_events: np.ndarray
    step_revision_eligible: np.ndarray
    step_active_count: np.ndarray
    step_masked_active_count: np.ndarray
    step_unresolved_count: np.ndarray

    summaries: list[StepSummary]


def _linear_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, len(values))
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0:
        return 0.0
    return float(np.dot(x, values - values.mean()) / denom)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _nonempty_concat(arrays: list[np.ndarray]) -> np.ndarray:
    usable = [np.asarray(a) for a in arrays if np.asarray(a).size]
    if not usable:
        return np.asarray([], dtype=float)
    return np.concatenate(usable).astype(float)


def features_from_payloads(
    payloads: list[TrajectoryFeaturePayload],
    late_fraction: float = 0.25,
) -> dict[str, float]:
    """Compute exact example-level features from ordered mergeable payloads.

    This is used both for whole-sequence diffusion and for blockwise SDAR.
    Unlike averaging block-level percentiles or slopes, it concatenates the
    underlying sufficient statistics before computing global upper tails,
    temporal trends, late-stage statistics, and decision-time features.
    """

    if not payloads:
        raise ValueError("No trajectory payloads to merge.")
    if not 0.0 < late_fraction <= 1.0:
        raise ValueError("late_fraction must be in (0, 1]")

    entropy = _nonempty_concat([p.entropy_samples for p in payloads])
    confidence = _nonempty_concat([p.confidence_samples for p in payloads])
    margin = _nonempty_concat([p.margin_samples for p in payloads])
    js = _nonempty_concat([p.js_samples for p in payloads])

    decision_entropy = _nonempty_concat([p.decision_entropy for p in payloads])
    decision_confidence = _nonempty_concat([p.decision_confidence for p in payloads])
    decision_margin = _nonempty_concat([p.decision_margin for p in payloads])
    commitment = _nonempty_concat([p.commitment_time for p in payloads])

    if not entropy.size or not decision_entropy.size or not commitment.size:
        raise ValueError("Trajectory payload is missing required predictive statistics.")

    step_entropy = _nonempty_concat([p.step_entropy for p in payloads])
    step_confidence = _nonempty_concat([p.step_confidence for p in payloads])
    step_margin = _nonempty_concat([p.step_margin for p in payloads])
    step_js = _nonempty_concat([p.step_js for p in payloads])
    step_revision_rate = _nonempty_concat([p.step_revision_rate for p in payloads])

    step_js_sum = _nonempty_concat([p.step_js_sum for p in payloads])
    step_js_count = _nonempty_concat([p.step_js_count for p in payloads])
    step_revision_events = _nonempty_concat([p.step_revision_events for p in payloads])
    step_revision_eligible = _nonempty_concat([p.step_revision_eligible for p in payloads])
    step_active_count = _nonempty_concat([p.step_active_count for p in payloads])
    step_masked_active_count = _nonempty_concat(
        [p.step_masked_active_count for p in payloads]
    )
    step_unresolved_count = _nonempty_concat([p.step_unresolved_count for p in payloads])

    n_steps = len(step_active_count)
    if n_steps == 0:
        raise ValueError("Trajectory payload contains no step statistics.")

    late_start = min(
        n_steps - 1,
        max(0, int(math.floor(n_steps * (1 - late_fraction)))),
    )

    valid_final = np.flatnonzero(step_unresolved_count > 0)
    if len(valid_final) == 0:
        raise ValueError("Trajectory payload has no unresolved observation.")
    final_i = int(valid_final[-1])

    valid_js_steps = np.flatnonzero(step_js_count > 0)
    js_final = (
        float(step_js[valid_js_steps[-1]])
        if len(valid_js_steps)
        else 0.0
    )

    late_js_count = float(step_js_count[late_start:].sum())
    late_js_mean = _ratio(
        float(step_js_sum[late_start:].sum()),
        late_js_count,
    )

    return {
        "trajectory_steps": float(n_steps),
        "entropy_mean": float(entropy.mean()),
        "entropy_p90": float(np.quantile(entropy, 0.90)),
        "entropy_max": float(entropy.max()),
        "entropy_final": float(step_entropy[final_i]),
        "entropy_slope": _linear_slope(step_entropy),
        "confidence_mean": float(confidence.mean()),
        "confidence_final": float(step_confidence[final_i]),
        "margin_mean": float(margin.mean()),
        "margin_min": float(margin.min()),
        "margin_final": float(step_margin[final_i]),
        "decision_entropy_mean": float(decision_entropy.mean()),
        "decision_entropy_p90": float(np.quantile(decision_entropy, 0.90)),
        "decision_confidence_mean": float(decision_confidence.mean()),
        "decision_margin_mean": float(decision_margin.mean()),
        "js_mean": float(js.mean()) if js.size else 0.0,
        "js_p90": float(np.quantile(js, 0.90)) if js.size else 0.0,
        "js_max": float(js.max()) if js.size else 0.0,
        "js_final": js_final,
        "js_slope": _linear_slope(step_js),
        "late_js_mean": late_js_mean,
        "revision_rate": _ratio(
            float(step_revision_events.sum()),
            float(step_revision_eligible.sum()),
        ),
        "late_revision_rate": _ratio(
            float(step_revision_events[late_start:].sum()),
            float(step_revision_eligible[late_start:].sum()),
        ),
        "revision_slope": _linear_slope(step_revision_rate),
        "mask_auc": _ratio(
            float(step_masked_active_count.sum()),
            float(step_active_count.sum()),
        ),
        "mask_fraction_final": _ratio(
            float(step_masked_active_count[final_i]),
            float(step_active_count[final_i]),
        ),
        "late_mask_fraction": _ratio(
            float(step_masked_active_count[late_start:].sum()),
            float(step_active_count[late_start:].sum()),
        ),
        "commitment_time_mean": float(commitment.mean()),
        "commitment_time_p90": float(np.quantile(commitment, 0.90)),
    }


def merge_trajectory_payloads(
    payloads: list[TrajectoryFeaturePayload],
    late_fraction: float = 0.25,
) -> tuple[dict[str, float], list[StepSummary]]:
    """Merge ordered block-local trajectories with exact global aggregation."""

    features = features_from_payloads(payloads, late_fraction=late_fraction)
    summaries: list[StepSummary] = []
    global_step = 0
    for payload in payloads:
        for summary in payload.summaries:
            summaries.append(replace(summary, step=global_step))
            global_step += 1
    return features, summaries


class TrajectoryAccumulator:
    """Collect uncertainty from positions that are actually being denoised.

    Schema v4 excludes prompt tokens, inactive future blocks, and already
    committed tokens from entropy/confidence/margin aggregation. Revision churn
    and JS drift are computed only when the same position is unresolved in two
    consecutive observations.

    The pilot deliberately supports batch size 1 so the telemetry contract is
    easy to audit before batched serving is introduced.
    """

    def __init__(self, prompt_length: int, mask_id: int | None, late_fraction: float = 0.25):
        if not 0.0 < late_fraction <= 1.0:
            raise ValueError("late_fraction must be in (0, 1]")
        self.prompt_length = int(prompt_length)
        self.mask_id = mask_id
        self.late_fraction = float(late_fraction)
        self._steps: list[_ObservedStep] = []
        self._previous_probabilities = None
        self._previous_unresolved = None

    def observe(self, step: int, logits, state_ids, active_position_mask=None) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to observe model trajectories.")
        if logits.ndim != 3 or state_ids.ndim != 2:
            raise ValueError("Expected logits [B,L,V] and state_ids [B,L].")
        if logits.shape[0] != 1 or state_ids.shape[0] != 1:
            raise ValueError("Schema v4 pilot supports batch size 1 only.")

        logits = logits[:, self.prompt_length :, :].detach()
        state = state_ids[:, self.prompt_length :].detach()

        if active_position_mask is None:
            active = torch.ones_like(state, dtype=torch.bool)
        else:
            active = active_position_mask.detach().bool()
            if active.ndim != 2 or active.shape[0] != 1:
                raise ValueError("active_position_mask must have shape [1,L].")
            if active.shape[1] == state_ids.shape[1]:
                active = active[:, self.prompt_length :]
            elif active.shape[1] != state.shape[1]:
                raise ValueError(
                    "active_position_mask must align with full state_ids or generated region."
                )

        if logits.shape[1] != state.shape[1]:
            if logits.shape[1] > state.shape[1]:
                logits = logits[:, -state.shape[1] :, :]
            else:
                state = state[:, -logits.shape[1] :]
                active = active[:, -logits.shape[1] :]

        if logits.shape[1] == 0:
            return

        logp = torch.log_softmax(logits.float(), dim=-1)
        p = logp.exp()
        vocab = logits.shape[-1]
        entropy = -(p * logp).sum(dim=-1) / max(math.log(vocab), 1.0)
        top2 = torch.topk(p, k=2, dim=-1)
        confidence = top2.values[..., 0]
        margin = top2.values[..., 0] - top2.values[..., 1]
        prediction = top2.indices[..., 0]

        if self.mask_id is None:
            masked = torch.ones_like(state, dtype=torch.bool)
        else:
            masked = state.eq(int(self.mask_id))

        unresolved = active & masked if self.mask_id is not None else active
        js_drift = torch.full_like(entropy, float("nan"))
        if (
            self._previous_probabilities is not None
            and self._previous_unresolved is not None
            and self._previous_probabilities.shape == p.shape
        ):
            eligible = unresolved & self._previous_unresolved
            if bool(eligible.any()):
                current_p = p[eligible]
                previous_p = self._previous_probabilities[eligible]
                mixture = 0.5 * (current_p + previous_p)
                eps = torch.finfo(current_p.dtype).tiny
                current_kl = (
                    current_p
                    * (
                        torch.log(current_p.clamp_min(eps))
                        - torch.log(mixture.clamp_min(eps))
                    )
                ).sum(dim=-1)
                previous_kl = (
                    previous_p
                    * (
                        torch.log(previous_p.clamp_min(eps))
                        - torch.log(mixture.clamp_min(eps))
                    )
                ).sum(dim=-1)
                js_values = 0.5 * (current_kl + previous_kl) / math.log(2.0)
                js_drift[eligible] = js_values

        self._previous_probabilities = p.detach()
        self._previous_unresolved = unresolved.detach()

        self._steps.append(
            _ObservedStep(
                step=int(step),
                entropy=entropy[0].cpu().numpy(),
                confidence=confidence[0].cpu().numpy(),
                margin=margin[0].cpu().numpy(),
                js_drift=js_drift[0].cpu().numpy(),
                masked=masked[0].cpu().numpy(),
                active=active[0].cpu().numpy(),
                prediction=prediction[0].cpu().numpy(),
                state=state[0].cpu().numpy(),
            )
        )

    def feature_payload(self) -> TrajectoryFeaturePayload:
        if not self._steps:
            raise ValueError("No trajectory steps were observed.")

        lengths = {len(s.entropy) for s in self._steps}
        if len(lengths) != 1:
            raise ValueError(
                "Observed generated length changed across steps. "
                "Use a runtime adapter that aligns logits/state to a fixed generated region."
            )

        ent = np.stack([s.entropy for s in self._steps])
        conf = np.stack([s.confidence for s in self._steps])
        margin = np.stack([s.margin for s in self._steps])
        js_drift = np.stack([s.js_drift for s in self._steps])
        masked = np.stack([s.masked for s in self._steps]).astype(bool)
        active = np.stack([s.active for s in self._steps]).astype(bool)
        pred = np.stack([s.prediction for s in self._steps])

        unresolved = active & masked if self.mask_id is not None else active
        if not unresolved.any():
            raise ValueError("No active unresolved positions were observed.")

        n_steps = len(self._steps)
        revision_events = np.zeros_like(unresolved, dtype=bool)
        revision_eligible = np.zeros_like(unresolved, dtype=bool)
        if n_steps > 1:
            revision_eligible[1:] = unresolved[1:] & unresolved[:-1]
            revision_events[1:] = revision_eligible[1:] & (pred[1:] != pred[:-1])

        js_valid = np.isfinite(js_drift) & unresolved

        step_entropy = np.full(n_steps, np.nan, dtype=float)
        step_confidence = np.full(n_steps, np.nan, dtype=float)
        step_margin = np.full(n_steps, np.nan, dtype=float)
        step_js = np.full(n_steps, np.nan, dtype=float)
        step_revision_rate = np.full(n_steps, np.nan, dtype=float)
        step_js_sum = np.zeros(n_steps, dtype=float)
        step_js_count = np.zeros(n_steps, dtype=float)
        step_revision_events = revision_events.sum(axis=1).astype(float)
        step_revision_eligible = revision_eligible.sum(axis=1).astype(float)
        step_active_count = active.sum(axis=1).astype(float)
        step_masked_active_count = (active & masked).sum(axis=1).astype(float)
        step_unresolved_count = unresolved.sum(axis=1).astype(float)

        summaries: list[StepSummary] = []
        for i, s in enumerate(self._steps):
            unresolved_i = unresolved[i]
            active_i = active[i]
            if unresolved_i.any():
                ent_i = ent[i][unresolved_i]
                conf_i = conf[i][unresolved_i]
                margin_i = margin[i][unresolved_i]
                step_entropy[i] = float(ent_i.mean())
                step_confidence[i] = float(conf_i.mean())
                step_margin[i] = float(margin_i.mean())
                entropy_p90 = float(np.quantile(ent_i, 0.90))
            else:
                entropy_p90 = 0.0

            if js_valid[i].any():
                js_values = js_drift[i][js_valid[i]].astype(float)
                step_js[i] = float(js_values.mean())
                step_js_sum[i] = float(js_values.sum())
                step_js_count[i] = float(len(js_values))

            if step_revision_eligible[i] > 0:
                step_revision_rate[i] = _ratio(
                    step_revision_events[i],
                    step_revision_eligible[i],
                )

            summaries.append(
                StepSummary(
                    step=s.step,
                    entropy_mean=(
                        float(step_entropy[i]) if np.isfinite(step_entropy[i]) else 0.0
                    ),
                    entropy_p90=entropy_p90,
                    confidence_mean=(
                        float(step_confidence[i])
                        if np.isfinite(step_confidence[i])
                        else 0.0
                    ),
                    margin_mean=(
                        float(step_margin[i]) if np.isfinite(step_margin[i]) else 0.0
                    ),
                    mask_fraction=_ratio(
                        float(step_masked_active_count[i]),
                        float(step_active_count[i]),
                    ),
                    revision_fraction=(
                        float(step_revision_rate[i])
                        if np.isfinite(step_revision_rate[i])
                        else 0.0
                    ),
                    js_mean=(
                        float(step_js[i]) if np.isfinite(step_js[i]) else 0.0
                    ),
                    active_fraction=float(active_i.mean()),
                    active_count=int(active_i.sum()),
                )
            )

        entropy_samples = ent[unresolved].astype(float)
        confidence_samples = conf[unresolved].astype(float)
        margin_samples = margin[unresolved].astype(float)
        js_samples = js_drift[js_valid].astype(float)

        decision_entropy: list[float] = []
        decision_confidence: list[float] = []
        decision_margin: list[float] = []
        commitment_time: list[float] = []

        for j in range(unresolved.shape[1]):
            unresolved_steps = np.flatnonzero(unresolved[:, j])
            active_steps = np.flatnonzero(active[:, j])
            if len(unresolved_steps):
                last = int(unresolved_steps[-1])
                decision_entropy.append(float(ent[last, j]))
                decision_confidence.append(float(conf[last, j]))
                decision_margin.append(float(margin[last, j]))

            if len(active_steps) == 0:
                continue
            unresolved_on_active = (
                masked[active_steps, j]
                if self.mask_id is not None
                else np.ones(len(active_steps), dtype=bool)
            )
            committed = np.flatnonzero(~unresolved_on_active)
            if len(committed):
                decision_index = max(0, int(committed[0]) - 1)
            else:
                decision_index = len(active_steps) - 1
            if len(active_steps) == 1:
                commitment_time.append(1.0)
            else:
                commitment_time.append(
                    decision_index / float(len(active_steps) - 1)
                )

        return TrajectoryFeaturePayload(
            entropy_samples=entropy_samples,
            confidence_samples=confidence_samples,
            margin_samples=margin_samples,
            js_samples=js_samples,
            decision_entropy=np.asarray(decision_entropy, dtype=float),
            decision_confidence=np.asarray(decision_confidence, dtype=float),
            decision_margin=np.asarray(decision_margin, dtype=float),
            commitment_time=np.asarray(commitment_time, dtype=float),
            step_entropy=step_entropy,
            step_confidence=step_confidence,
            step_margin=step_margin,
            step_js=step_js,
            step_revision_rate=step_revision_rate,
            step_js_sum=step_js_sum,
            step_js_count=step_js_count,
            step_revision_events=step_revision_events,
            step_revision_eligible=step_revision_eligible,
            step_active_count=step_active_count,
            step_masked_active_count=step_masked_active_count,
            step_unresolved_count=step_unresolved_count,
            summaries=summaries,
        )

    def finalize(self) -> tuple[dict[str, float], list[StepSummary]]:
        payload = self.feature_payload()
        features = features_from_payloads(
            [payload],
            late_fraction=self.late_fraction,
        )
        return features, payload.summaries
