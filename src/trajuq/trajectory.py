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
    prediction: np.ndarray
    state: np.ndarray


def _linear_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, len(values))
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0:
        return 0.0
    return float(np.dot(x, values - values.mean()) / denom)


class TrajectoryAccumulator:
    """Collect compact uncertainty summaries from iterative denoising.

    Schema v1 deliberately supports batch size 1 so the telemetry contract is
    easy to audit before batched serving is introduced.
    """

    def __init__(self, prompt_length: int, mask_id: int | None, late_fraction: float = 0.25):
        if not 0.0 < late_fraction <= 1.0:
            raise ValueError("late_fraction must be in (0, 1]")
        self.prompt_length = int(prompt_length)
        self.mask_id = mask_id
        self.late_fraction = float(late_fraction)
        self._steps: list[_ObservedStep] = []

    def observe(self, step: int, logits, state_ids) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to observe model trajectories.")
        if logits.ndim != 3 or state_ids.ndim != 2:
            raise ValueError("Expected logits [B,L,V] and state_ids [B,L].")
        if logits.shape[0] != 1 or state_ids.shape[0] != 1:
            raise ValueError("Schema v1 pilot supports batch size 1 only.")

        logits = logits[:, self.prompt_length :, :].detach()
        state = state_ids[:, self.prompt_length :].detach()

        if logits.shape[1] != state.shape[1]:
            if logits.shape[1] > state.shape[1]:
                logits = logits[:, -state.shape[1] :, :]
            else:
                state = state[:, -logits.shape[1] :]

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
            masked = torch.zeros_like(state, dtype=torch.bool)
        else:
            masked = state.eq(int(self.mask_id))

        self._steps.append(
            _ObservedStep(
                step=int(step),
                entropy=entropy[0].cpu().numpy(),
                confidence=confidence[0].cpu().numpy(),
                margin=margin[0].cpu().numpy(),
                masked=masked[0].cpu().numpy(),
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
        masked = np.stack([s.masked for s in self._steps]).astype(float)
        pred = np.stack([s.prediction for s in self._steps])

        revision = np.zeros_like(masked, dtype=float)
        if len(self._steps) > 1:
            revision[1:] = (pred[1:] != pred[:-1]).astype(float)

        n_steps = len(self._steps)
        late_start = min(
            n_steps - 1,
            max(0, int(math.floor(n_steps * (1 - self.late_fraction)))),
        )

        first_commit = np.full(masked.shape[1], n_steps - 1, dtype=float)
        if self.mask_id is not None:
            for j in range(masked.shape[1]):
                idx = np.where(masked[:, j] == 0)[0]
                if len(idx):
                    first_commit[j] = float(idx[0])
        first_commit /= max(n_steps - 1, 1)

        step_entropy = ent.mean(axis=1)
        step_revision = revision.mean(axis=1)

        features = {
            "trajectory_steps": float(n_steps),
            "entropy_mean": float(ent.mean()),
            "entropy_p90": float(np.quantile(ent, 0.90)),
            "entropy_max": float(ent.max()),
            "entropy_final": float(ent[-1].mean()),
            "entropy_slope": _linear_slope(step_entropy),
            "confidence_mean": float(conf.mean()),
            "confidence_final": float(conf[-1].mean()),
            "margin_mean": float(margin.mean()),
            "margin_min": float(margin.min()),
            "margin_final": float(margin[-1].mean()),
            "revision_rate": float(revision[1:].mean()) if n_steps > 1 else 0.0,
            "late_revision_rate": float(revision[late_start:].mean()),
            "revision_slope": _linear_slope(step_revision),
            "mask_auc": float(masked.mean()),
            "mask_fraction_final": float(masked[-1].mean()),
            "late_mask_fraction": float(masked[late_start:].mean()),
            "commitment_time_mean": float(first_commit.mean()),
            "commitment_time_p90": float(np.quantile(first_commit, 0.90)),
        }

        summaries = [
            StepSummary(
                step=s.step,
                entropy_mean=float(s.entropy.mean()),
                entropy_p90=float(np.quantile(s.entropy, 0.90)),
                confidence_mean=float(s.confidence.mean()),
                margin_mean=float(s.margin.mean()),
                mask_fraction=float(s.masked.mean()),
                revision_fraction=float(revision[i].mean()),
            )
            for i, s in enumerate(self._steps)
        ]
        return features, summaries
