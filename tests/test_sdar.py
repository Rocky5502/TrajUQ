from trajuq.adapters.sdar import _combine_block_features
from trajuq.schema import StepSummary


def _summary(step: int) -> StepSummary:
    return StepSummary(
        step=step,
        entropy_mean=0.2,
        entropy_p90=0.3,
        confidence_mean=0.8,
        margin_mean=0.6,
        mask_fraction=0.2,
        revision_fraction=0.1,
    )


def test_combine_sdar_block_features_weighted_and_extremal():
    a = {
        "trajectory_steps": 4.0,
        "entropy_mean": 0.2,
        "entropy_p90": 0.3,
        "entropy_max": 0.4,
        "entropy_final": 0.1,
        "entropy_slope": -0.1,
        "confidence_mean": 0.8,
        "confidence_final": 0.9,
        "margin_mean": 0.7,
        "margin_min": 0.5,
        "margin_final": 0.8,
        "revision_rate": 0.1,
        "late_revision_rate": 0.05,
        "revision_slope": -0.02,
        "mask_auc": 0.4,
        "mask_fraction_final": 0.0,
        "late_mask_fraction": 0.1,
        "commitment_time_mean": 0.3,
        "commitment_time_p90": 0.5,
    }
    b = dict(a)
    b.update({"entropy_mean": 0.8, "entropy_max": 0.9, "margin_min": 0.2})
    features, summaries = _combine_block_features(
        [(1, a, [_summary(0)]), (3, b, [_summary(0), _summary(1)])]
    )
    assert abs(features["entropy_mean"] - 0.65) < 1e-12
    assert features["entropy_max"] == 0.9
    assert features["margin_min"] == 0.2
    assert features["trajectory_steps"] == 8.0
    assert [s.step for s in summaries] == [0, 1, 2]
