from __future__ import annotations


# Fair final-decision baseline: mean normalized entropy immediately before each
# generated token's commitment decision.
FINAL_BASELINE_FEATURES = ["decision_entropy_mean"]

# Nested primary model: it contains the exact baseline feature plus frozen
# trajectory-level information. Any held-out gain therefore tests incremental
# information beyond the final per-token commitment confidence.
PRIMARY_TRAJECTORY_FEATURES = [
    "decision_entropy_mean",
    "entropy_mean",
    "entropy_p90",
    "entropy_slope",
    "margin_mean",
    "margin_min",
    "js_mean",
    "js_p90",
    "js_slope",
    "late_js_mean",
    "revision_rate",
    "late_revision_rate",
    "mask_auc",
    "late_mask_fraction",
    "commitment_time_mean",
]


DISTRIBUTION_TRAJECTORY_FEATURES = [
    "entropy_mean",
    "entropy_p90",
    "entropy_slope",
    "margin_mean",
    "margin_min",
]

STABILITY_FEATURES = [
    "js_mean",
    "js_p90",
    "js_slope",
    "late_js_mean",
    "revision_rate",
    "late_revision_rate",
]

PERSISTENCE_FEATURES = [
    "mask_auc",
    "late_mask_fraction",
    "commitment_time_mean",
]

FROZEN_FEATURE_GROUPS = {
    "decision_only": FINAL_BASELINE_FEATURES,
    "decision_plus_distribution": FINAL_BASELINE_FEATURES + DISTRIBUTION_TRAJECTORY_FEATURES,
    "decision_plus_stability": FINAL_BASELINE_FEATURES + STABILITY_FEATURES,
    "decision_plus_persistence": FINAL_BASELINE_FEATURES + PERSISTENCE_FEATURES,
    "full": PRIMARY_TRAJECTORY_FEATURES,
    "full_without_js": [
        name
        for name in PRIMARY_TRAJECTORY_FEATURES
        if name not in {"js_mean", "js_p90", "js_slope", "late_js_mean"}
    ],
    "full_without_revisions": [
        name
        for name in PRIMARY_TRAJECTORY_FEATURES
        if name not in {"revision_rate", "late_revision_rate"}
    ],
    "full_without_persistence": [
        name
        for name in PRIMARY_TRAJECTORY_FEATURES
        if name not in {"mask_auc", "late_mask_fraction", "commitment_time_mean"}
    ],
}
