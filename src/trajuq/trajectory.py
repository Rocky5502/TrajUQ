from __future__ import annotations

import math
from dataclasses import dataclass

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
    masked: np.ndarray
    active: np.ndarray
    prediction: np.ndarray
    state: np.ndarray


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


def _masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    if selected.size == 0:
        raise ValueError("Trajectory summary has no active unresolved positions.")
    return selected.astype(float)


def _safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> float:
    denom = int(np.asarray(denominator, dtype=bool).sum())
    if denom == 0:
        return 0.0
    return float(np.asarray(numerator, dtype=bool).sum() / denom)


class TrajectoryAccumulator:
    """Collect uncertainty from positions that are actually being denoised.

    Schema v2 excludes prompt tokens, inactive future blocks, and already
    committed tokens from entropy/confidence/margin aggregation. Revision churn
    is counted only when the same position is unresolved in two consecutive
    observations. This avoids treating irrelevant logits on committed tokens as
    uncertainty evidence.

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

    def observe(self, step: int, logits, state_ids, active_position_mask=None) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to observe model trajectories.")
        if logits.ndim != 3 or state_ids.ndim != 2:
            raise ValueError("Expected logits [B,L,V] and state_ids [B,L].")
        if logits.shape[0] != 1 or state_ids.shape[0] != 1:
            raise ValueError("Schema v2 pilot supports batch size 1 only.")

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

        self._steps.append(
            _ObservedStep(
                step=int(step),
                entropy=entropy[0].cpu().numpy(),
                confidence=confidence[0].cpu().numpy(),
                margin=margin[0].cpu().numpy(),
                masked=masked[0].cpu().numpy(),
                active=active[0].cpu().numpy(),
                prediction=prediction[0].cpu().numpy(),
                state=state[0].cpu().numpy(),
            )
        )

    def finalize(self) -> tuple[dict[str, float], list[StepSummary]]:
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
        masked = np.stack([s.masked for s in self._steps]).astype(bool)
        active = np.stack([s.active for s in self._steps]).astype(bool)
        pred = np.stack([s.prediction for s in self._steps])

        unresolved = active & masked if self.mask_id is not None else active
        if not unresolved.any():
            raise ValueError("No active unresolved positions were observed.")

        n_steps = len(self._steps)
        late_start = min(
            n_steps - 1,
            max(0, int(math.floor(n_steps * (1 - self.late_fraction)))),
        )

        revision_events = np.zeros_like(unresolved, dtype=bool)
        revision_eligible = np.zeros_like(unresolved, dtype=bool)
        if n_steps > 1:
            revision_eligible[1:] = unresolved[1:] & unresolved[:-1]
            revision_events[1:] = revision_eligible[1:] & (pred[1:] != pred[:-1])

        step_entropy = np.full(n_steps, np.nan, dtype=float)
        step_revision = np.full(n_steps, np.nan, dtype=float)
        for i in range(n_steps):
            if unresolved[i].any():
                step_entropy[i] = float(ent[i][unresolved[i]].mean())
            if revision_eligible[i].any():
                step_revision[i] = _safe_fraction(
                    revision_events[i],
                    revision_eligible[i],
                )

        valid_final_steps = np.flatnonzero(unresolved.any(axis=1))
        final_i = int(valid_final_steps[-1])
        final_mask = unresolved[final_i]

        active_ent = _masked_values(ent, unresolved)
        active_conf = _masked_values(conf, unresolved)
        active_margin = _masked_values(margin, unresolved)

        late_unresolved = unresolved[late_start:]
        late_active = active[late_start:]
        late_masked = active[late_start:] & masked[late_start:]

        commitment_times: list[float] = []
        for j in range(active.shape[1]):
            active_steps = np.flatnonzero(active[:, j])
            if len(active_steps) == 0:
                continue
            unresolved_on_active = masked[active_steps, j] if self.mask_id is not None else np.ones(
                len(active_steps),
                dtype=bool,
            )
            committed = np.flatnonzero(~unresolved_on_active)
            if len(committed):
                local_index = int(committed[0])
            else:
                # The last transfer happens after the last pre-transfer
                # observation, so an unresolved final observation means
                # commitment at the end of the active window.
                local_index = len(active_steps) - 1
            if len(active_steps) == 1:
                commitment_times.append(1.0)
            else:
                commitment_times.append(local_index / float(len(active_steps) - 1))

        if not commitment_times:
            raise ValueError("No active positions available for commitment-time features.")
        commitment = np.asarray(commitment_times, dtype=float)

        features = {
            "trajectory_steps": float(n_steps),
            "entropy_mean": float(active_ent.mean()),
            "entropy_p90": float(np.quantile(active_ent, 0.90)),
            "entropy_max": float(active_ent.max()),
            "entropy_final": float(ent[final_i][final_mask].mean()),
            "entropy_slope": _linear_slope(step_entropy),
            "confidence_mean": float(active_conf.mean()),
            "confidence_final": float(conf[final_i][final_mask].mean()),
            "margin_mean": float(active_margin.mean()),
            "margin_min": float(active_margin.min()),
            "margin_final": float(margin[final_i][final_mask].mean()),
            "revision_rate": _safe_fraction(revision_events, revision_eligible),
            "late_revision_rate": _safe_fraction(
                revision_events[late_start:],
                revision_eligible[late_start:],
            ),
            "revision_slope": _linear_slope(step_revision),
            "mask_auc": _safe_fraction(active & masked, active),
            "mask_fraction_final": _safe_fraction(
                active[final_i] & masked[final_i],
                active[final_i],
            ),
            "late_mask_fraction": _safe_fraction(late_masked, late_active),
            "commitment_time_mean": float(commitment.mean()),
            "commitment_time_p90": float(np.quantile(commitment, 0.90)),
        }

        summaries: list[StepSummary] = []
        for i, s in enumerate(self._steps):
            unresolved_i = unresolved[i]
            active_i = active[i]
            if unresolved_i.any():
                ent_i = ent[i][unresolved_i]
                conf_i = conf[i][unresolved_i]
                margin_i = margin[i][unresolved_i]
                entropy_mean = float(ent_i.mean())
                entropy_p90 = float(np.quantile(ent_i, 0.90))
                confidence_mean = float(conf_i.mean())
                margin_mean = float(margin_i.mean())
            else:
                entropy_mean = 0.0
                entropy_p90 = 0.0
                confidence_mean = 0.0
                margin_mean = 0.0

            summaries.append(
                StepSummary(
                    step=s.step,
                    entropy_mean=entropy_mean,
                    entropy_p90=entropy_p90,
                    confidence_mean=confidence_mean,
                    margin_mean=margin_mean,
                    mask_fraction=_safe_fraction(active_i & masked[i], active_i),
                    revision_fraction=_safe_fraction(
                        revision_events[i],
                        revision_eligible[i],
                    ),
                    active_fraction=float(active_i.mean()),
                    active_count=int(active_i.sum()),
                )
            )
        return features, summaries
