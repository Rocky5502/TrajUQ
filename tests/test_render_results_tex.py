import pytest

from scripts.render_results_tex import (
    ABLATION_SUFFIXES,
    PRIOR_SIGNAL_SUFFIXES,
    build_macros,
)


def primary():
    metrics = {}
    for name, baseline, candidate in [
        ("auroc", 0.60, 0.70),
        ("auprc", 0.55, 0.66),
        ("brier", 0.24, 0.20),
        ("ece15", 0.12, 0.08),
        ("aurc", 0.30, 0.22),
    ]:
        metrics[name] = {
            "baseline_macro": baseline,
            "candidate_macro": candidate,
            "improvement_macro": (
                candidate - baseline
                if name in {"auroc", "auprc"}
                else baseline - candidate
            ),
            "improvement_ci_low": 0.01,
            "improvement_ci_high": 0.10,
        }
    metrics["auroc"]["positive_model_families"] = 3
    metrics["auroc"]["n_model_families"] = 3
    return {
        "protocol": "hierarchical_within_cell_macro_bootstrap_v1",
        "cell_set_complete": True,
        "expected_cell_count": 18,
        "observed_cell_count": 18,
        "macro": metrics,
    }


def claim(supported: bool = True):
    return {
        "protocol": "trajuq_rq1_claim_gate_v1",
        "decision": (
            "SUPPORT_POSITIVE_RQ1_CLAIM"
            if supported
            else "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM"
        ),
        "claim_supported": supported,
        "observed": {
            "macro_auroc_improvement": 0.10,
            "macro_auroc_ci_low": 0.01,
            "macro_auroc_ci_high": 0.10,
            "positive_model_families": 3,
            "n_model_families": 3,
        },
    }


def repeated():
    return {
        "protocol": "trajuq_vs_repeated_macro_v2",
        "comparison": "single_trajectory_trajuq_vs_five_sample_baseline",
        "cell_set_complete": True,
        "expected_cells": 18,
        "observed_cells": 18,
        "auxiliary_compute": {
            "aggregation": "equal_cell_weight_macro",
            "expected_cells": 18,
            "observed_cells_with_complete_test_wall_time": 18,
            "complete": True,
            "sample_count": 5,
            "mean_test_auxiliary_wall_time_seconds_per_example": 12.34,
            "mean_test_p90_auxiliary_wall_time_seconds_per_example": 15.67,
        },
        "macro": {
            name: {
                "baseline_macro": baseline,
                "candidate_macro": candidate,
            }
            for name, baseline, candidate in [
                ("auroc", 0.62, 0.70),
                ("auprc", 0.58, 0.66),
                ("aurc", 0.28, 0.22),
            ]
        },
    }


def ablation():
    groups = {}
    for idx, name in enumerate(ABLATION_SUFFIXES, start=1):
        delta = 0.01 * idx
        groups[name] = {
            "metrics": {
                metric: {
                    "baseline_macro": 0.60,
                    "candidate_macro": 0.60 + delta,
                    "improvement_macro": delta,
                    "improvement_ci_low": delta - 0.01,
                    "improvement_ci_high": delta + 0.01,
                }
                for metric in ("auroc", "auprc", "aurc")
            }
        }
    return {
        "protocol": "trajuq_feature_ablation_macro_v1",
        "baseline": "decision_only",
        "cell_set_complete": True,
        "expected_cells": 18,
        "observed_cells": 18,
        "groups": groups,
    }


def prior_signals():
    return {
        "protocol": "trajuq_prior_aligned_signal_macro_v1",
        "role": "secondary_prior_aligned_signal_class_diagnostic",
        "cell_set_complete": True,
        "expected_cells": 18,
        "observed_cells": 18,
        "macro": {
            name: {
                metric: {
                    "baseline_macro": 0.61,
                    "candidate_macro": 0.70,
                    "improvement_macro": 0.09,
                    "improvement_ci_low": 0.02,
                    "improvement_ci_high": 0.15,
                }
                for metric in ("auroc", "auprc", "aurc")
            }
            for name in PRIOR_SIGNAL_SUFFIXES
        },
    }


def adaptive():
    return {
        "protocol": "trajuq_adaptive_macro_v1",
        "execution_semantics": "selective_second_pass_rerun",
        "fraction_key": "cap_050",
        "cell_set_complete": True,
        "expected_cells": 18,
        "observed_cells": 18,
        "is_primary_fraction": True,
        "max_refine_fraction": 0.50,
        "macro": {
            "adaptive_accuracy": 0.75,
            "base_accuracy": 0.68,
            "refined_accuracy": 0.77,
            "random_expected_accuracy": 0.71,
            "delta_vs_random": 0.04,
            "nominal_cost_ratio_to_base": 2.8,
            "refined_cost_ratio_to_base": 3.3,
        },
    }


def test_result_macros_are_derived_from_complete_artifacts():
    macros = build_macros(
        primary(),
        claim(),
        repeated(),
        ablation(),
        prior_signals(),
        adaptive(),
    )
    assert macros["PrimaryTrajAUROC"] == "0.700"
    assert macros["RepeatedAUROC"] == "0.620"
    assert macros["RepeatedAuxWallTime"] == "12.34"
    assert macros["AblationFullAUROCDelta"] == "+0.040"
    assert macros["PriorEntropyAUROC"] == "0.610"
    assert macros["PriorEntropyAUROCDelta"] == "+0.090"
    assert macros["AdaptiveDeltaRandom"] == "+0.040"
    assert "positive point improvement" in macros["AbstractResultSentence"]
    assert macros["PaperStateLimitationsSentence"] == ""


def test_result_renderer_rejects_partial_primary_matrix():
    payload = primary()
    payload["cell_set_complete"] = False
    with pytest.raises(ValueError):
        build_macros(
            payload,
            claim(),
            repeated(),
            ablation(),
            prior_signals(),
            adaptive(),
        )


def test_result_renderer_uses_narrow_language_when_claim_gate_fails():
    macros = build_macros(
        primary(),
        claim(False),
        repeated(),
        ablation(),
        prior_signals(),
        adaptive(),
    )
    assert "not met" in macros["AbstractResultSentence"]


def test_result_renderer_rejects_contradictory_claim_flag_and_decision():
    payload = claim(True)
    payload["decision"] = "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM"
    with pytest.raises(ValueError, match="contradicts"):
        build_macros(
            primary(),
            payload,
            repeated(),
            ablation(),
            prior_signals(),
            adaptive(),
        )


def test_result_renderer_rejects_incomplete_repeated_compute():
    payload = repeated()
    payload["auxiliary_compute"]["complete"] = False
    with pytest.raises(ValueError, match="compute accounting is incomplete"):
        build_macros(
            primary(),
            claim(),
            payload,
            ablation(),
            prior_signals(),
            adaptive(),
        )


def test_result_renderer_rejects_incomplete_ablation_matrix():
    payload = ablation()
    payload["cell_set_complete"] = False
    with pytest.raises(ValueError, match="Feature-ablation macro cell set is incomplete"):
        build_macros(
            primary(),
            claim(),
            repeated(),
            payload,
            prior_signals(),
            adaptive(),
        )


def test_result_renderer_rejects_incomplete_prior_signal_matrix():
    payload = prior_signals()
    payload["cell_set_complete"] = False
    with pytest.raises(
        ValueError,
        match="Prior-signal macro cell set is incomplete",
    ):
        build_macros(
            primary(),
            claim(),
            repeated(),
            ablation(),
            payload,
            adaptive(),
        )
