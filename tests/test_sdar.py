import numpy as np
import pytest

from trajuq.adapters.sdar import bounded_transfer_count
from trajuq.schema import StepSummary
from trajuq.trajectory import TrajectoryFeaturePayload, merge_trajectory_payloads


def payload(
    entropy_samples,
    js_samples,
    decision_entropy,
    step_entropy,
    step_js,
    step_revision_events,
    step_revision_eligible,
):
    n = len(step_entropy)
    summaries = [
        StepSummary(
            step=i,
            entropy_mean=float(step_entropy[i]),
            entropy_p90=float(step_entropy[i]),
            confidence_mean=0.7,
            margin_mean=0.4,
            mask_fraction=0.5,
            revision_fraction=(
                float(step_revision_events[i] / step_revision_eligible[i])
                if step_revision_eligible[i]
                else 0.0
            ),
        )
        for i in range(n)
    ]
    js_arr = np.asarray(js_samples, dtype=float)
    js_mean = float(js_arr.mean()) if len(js_arr) else np.nan
    return TrajectoryFeaturePayload(
        entropy_samples=np.asarray(entropy_samples, dtype=float),
        confidence_samples=np.full(len(entropy_samples), 0.7),
        margin_samples=np.full(len(entropy_samples), 0.4),
        js_samples=js_arr,
        decision_entropy=np.asarray(decision_entropy, dtype=float),
        decision_confidence=np.full(len(decision_entropy), 0.7),
        decision_margin=np.full(len(decision_entropy), 0.4),
        commitment_time=np.full(len(decision_entropy), 0.5),
        step_entropy=np.asarray(step_entropy, dtype=float),
        step_confidence=np.full(n, 0.7),
        step_margin=np.full(n, 0.4),
        step_js=np.full(n, js_mean),
        step_revision_rate=np.asarray(
            [
                e / d if d else np.nan
                for e, d in zip(step_revision_events, step_revision_eligible)
            ],
            dtype=float,
        ),
        step_js_sum=np.full(n, js_arr.sum() / n if n else 0.0),
        step_js_count=np.full(n, len(js_arr) / n if n else 0.0),
        step_revision_events=np.asarray(step_revision_events, dtype=float),
        step_revision_eligible=np.asarray(step_revision_eligible, dtype=float),
        step_active_count=np.full(n, 2.0),
        step_masked_active_count=np.full(n, 1.0),
        step_unresolved_count=np.full(n, 1.0),
        summaries=summaries,
    )


def test_merge_block_payloads_recomputes_global_percentiles_and_slopes():
    a = payload(
        entropy_samples=[0.1, 0.2],
        js_samples=[0.05, 0.10],
        decision_entropy=[0.15],
        step_entropy=[0.2, 0.3],
        step_js=[0.05, 0.10],
        step_revision_events=[0, 1],
        step_revision_eligible=[0, 2],
    )
    b = payload(
        entropy_samples=[0.8, 0.9],
        js_samples=[0.6, 0.8],
        decision_entropy=[0.85, 0.75, 0.65],
        step_entropy=[0.7, 0.9],
        step_js=[0.6, 0.8],
        step_revision_events=[1, 1],
        step_revision_eligible=[2, 2],
    )

    features, summaries = merge_trajectory_payloads([a, b])

    assert np.isclose(
        features["entropy_p90"],
        np.quantile(np.array([0.1, 0.2, 0.8, 0.9]), 0.90),
    )
    assert np.isclose(
        features["decision_entropy_mean"],
        np.mean([0.15, 0.85, 0.75, 0.65]),
    )
    assert features["trajectory_steps"] == 4.0
    assert [s.step for s in summaries] == [0, 1, 2, 3]
    assert features["entropy_slope"] > 0


def test_reduced_step_transfer_count_never_exceeds_remaining_masks():
    assert bounded_transfer_count(2, 1) == 1
    assert bounded_transfer_count(2, 2) == 2
    assert bounded_transfer_count(2, 3) == 2
    assert bounded_transfer_count(1, 0) == 0


def test_transfer_count_rejects_negative_values():
    with pytest.raises(ValueError):
        bounded_transfer_count(-1, 2)
    with pytest.raises(ValueError):
        bounded_transfer_count(1, -2)
