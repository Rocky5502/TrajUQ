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
    assert np.isclose(features["revision_rate"], 1 / 3)
    assert summaries[0].active_count == 1
    assert summaries[1].active_count == 2
