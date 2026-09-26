from __future__ import annotations


FINAL_BASELINE_FEATURES = ["entropy_final"]

PRIMARY_TRAJECTORY_FEATURES = [
    "entropy_final",
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
