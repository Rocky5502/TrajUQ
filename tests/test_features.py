from pathlib import Path

import yaml

from trajuq.features import (
    FROZEN_FEATURE_GROUPS,
    FINAL_BASELINE_FEATURES,
    PRIMARY_TRAJECTORY_FEATURES,
)



def test_frozen_feature_groups_are_nested_on_decision_baseline():
    baseline = set(FINAL_BASELINE_FEATURES)
    assert FROZEN_FEATURE_GROUPS["decision_only"] == FINAL_BASELINE_FEATURES
    for name, features in FROZEN_FEATURE_GROUPS.items():
        assert len(features) == len(set(features)), name
        assert baseline.issubset(set(features)), name


def test_leave_family_out_groups_are_strict_ablations():
    full = set(FROZEN_FEATURE_GROUPS["full"])
    for name in ("full_without_js", "full_without_revisions", "full_without_persistence"):
        group = set(FROZEN_FEATURE_GROUPS[name])
        assert group < full


def test_primary_feature_contract_matches_frozen_config():
    cfg = yaml.safe_load(
        Path("configs/experiments/primary.yaml").read_text(encoding="utf-8")
    )

    assert FINAL_BASELINE_FEATURES == ["decision_entropy_mean"]
    assert PRIMARY_TRAJECTORY_FEATURES == cfg["trajectory"]["primary_features"]
    assert cfg["calibration"]["primary"] == "logistic"
    assert set(FROZEN_FEATURE_GROUPS) == set(
        cfg["feature_ablation"]["frozen_groups"]
    )
    assert (
        FROZEN_FEATURE_GROUPS["decision_only"]
        == FINAL_BASELINE_FEATURES
    )
    assert (
        FROZEN_FEATURE_GROUPS["full"]
        == PRIMARY_TRAJECTORY_FEATURES
    )
