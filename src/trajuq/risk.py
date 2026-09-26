from __future__ import annotations


RAW_RISK_VERSION = "trajuq-raw-risk-v4"


def raw_trajectory_risk(features: dict[str, float]) -> float:
    """Transparent, training-free trajectory ranking score.

    Inputs are normalized/ratio features from TrajectoryAccumulator. The score is
    deliberately simple and frozen before primary results. It is *not* a
    probability; use a held-out calibrator for error probabilities.
    """

    entropy_p90 = float(features["entropy_p90"])
    entropy_mean = float(features["entropy_mean"])
    margin_mean = float(features["margin_mean"])
    late_revision = float(features["late_revision_rate"])
    js_p90 = float(features.get("js_p90", 0.0))
    mask_auc = float(features["mask_auc"])
    commit = float(features["commitment_time_mean"])
    late_mask = float(features["late_mask_fraction"])

    return float(
        0.20 * entropy_p90
        + 0.10 * entropy_mean
        + 0.15 * (1.0 - margin_mean)
        + 0.15 * late_revision
        + 0.15 * js_p90
        + 0.10 * mask_auc
        + 0.10 * commit
        + 0.05 * late_mask
    )
