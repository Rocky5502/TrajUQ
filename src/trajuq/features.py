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
