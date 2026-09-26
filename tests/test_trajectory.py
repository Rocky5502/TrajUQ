import numpy as np

from trajuq.trajectory import TrajectoryAccumulator, _ObservedStep


def test_finalize_ignores_committed_and_inactive_logits():
    acc = TrajectoryAccumulator(prompt_length=0, mask_id=99)

    # Two positions. Position 0 is active throughout and commits after the
    # second observation. Position 1 is inactive until the second observation.
    acc._steps = [
        _ObservedStep(
            step=0,
            entropy=np.array([0.2, 0.99]),
            confidence=np.array([0.8, 0.01]),
            margin=np.array([0.6, 0.01]),
            js_drift=np.array([np.nan, np.nan]),
            masked=np.array([True, True]),
            active=np.array([True, False]),
            prediction=np.array([1, 7]),
            state=np.array([99, 99]),
        ),
        _ObservedStep(
            step=1,
            entropy=np.array([0.4, 0.6]),
            confidence=np.array([0.6, 0.4]),
            margin=np.array([0.3, 0.2]),
            js_drift=np.array([0.10, np.nan]),
            masked=np.array([True, True]),
            active=np.array([True, True]),
            prediction=np.array([2, 8]),
            state=np.array([99, 99]),
        ),
        _ObservedStep(
            step=2,
            entropy=np.array([0.95, 0.5]),
            confidence=np.array([0.02, 0.5]),
            margin=np.array([0.01, 0.25]),
            js_drift=np.array([np.nan, 0.20]),
            masked=np.array([False, True]),
            active=np.array([True, True]),
            prediction=np.array([9, 8]),
            state=np.array([1, 99]),
        ),
    ]

    features, summaries = acc.finalize()

    # The very uncertain committed-token logits at step 2 position 0 are excluded.
    assert features["entropy_max"] == 0.6
    # Revision at position 0 from step 0->1 counts; step 1->2 does not because
    # position 0 is already committed at the step-2 observation.
    assert np.isclose(features["revision_rate"], 0.5)
    # Position 0 commits after step 1 (entropy 0.4); position 1 commits after
    # the final observed step (entropy 0.5).
    assert np.isclose(features["decision_entropy_mean"], 0.45)
    assert np.isclose(features["commitment_time_mean"], 0.75)
    assert summaries[0].active_count == 1
    assert summaries[1].active_count == 2


def test_js_drift_uses_only_consecutive_unresolved_positions():
    acc = TrajectoryAccumulator(prompt_length=0, mask_id=99)
    acc._steps = [
        _ObservedStep(
            step=0,
            entropy=np.array([0.2]),
            confidence=np.array([0.8]),
            margin=np.array([0.6]),
            js_drift=np.array([np.nan]),
            masked=np.array([True]),
            active=np.array([True]),
            prediction=np.array([1]),
            state=np.array([99]),
        ),
        _ObservedStep(
            step=1,
            entropy=np.array([0.3]),
            confidence=np.array([0.7]),
            margin=np.array([0.5]),
            js_drift=np.array([0.25]),
            masked=np.array([True]),
            active=np.array([True]),
            prediction=np.array([1]),
            state=np.array([99]),
        ),
    ]
    features, _ = acc.finalize()
    assert np.isclose(features["js_mean"], 0.25)
    assert np.isclose(features["js_p90"], 0.25)
