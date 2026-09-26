from trajuq.risk import raw_trajectory_risk


def test_raw_risk_increases_with_instability():
    calm = {
        "entropy_p90": 0.1,
        "entropy_mean": 0.1,
        "margin_mean": 0.9,
        "late_revision_rate": 0.0,
        "js_p90": 0.02,
        "mask_auc": 0.1,
        "commitment_time_mean": 0.1,
        "late_mask_fraction": 0.0,
    }
    unstable = {
        "entropy_p90": 0.8,
        "entropy_mean": 0.7,
        "margin_mean": 0.2,
        "late_revision_rate": 0.6,
        "js_p90": 0.7,
        "mask_auc": 0.7,
        "commitment_time_mean": 0.8,
        "late_mask_fraction": 0.5,
    }
    assert raw_trajectory_risk(unstable) > raw_trajectory_risk(calm)
