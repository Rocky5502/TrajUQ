import pytest

from scripts.render_results_tex import build_macros


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
        "cell_set_complete": True,
        "expected_cells": 18,
        "observed_cells": 18,
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


def adaptive():
    return {
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
    macros = build_macros(primary(), claim(), repeated(), adaptive())
    assert macros["PrimaryTrajAUROC"] == "0.700"
    assert macros["RepeatedAUROC"] == "0.620"
    assert macros["AdaptiveDeltaRandom"] == "+0.040"
    assert "positive point improvement" in macros["AbstractResultSentence"]


def test_result_renderer_rejects_partial_primary_matrix():
    payload = primary()
    payload["cell_set_complete"] = False
    with pytest.raises(ValueError):
        build_macros(payload, claim(), repeated(), adaptive())


def test_result_renderer_uses_narrow_language_when_claim_gate_fails():
    macros = build_macros(primary(), claim(False), repeated(), adaptive())
    assert "not met" in macros["AbstractResultSentence"]


def test_result_renderer_rejects_contradictory_claim_flag_and_decision():
    payload = claim(True)
    payload["decision"] = "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM"
    with pytest.raises(ValueError, match="contradicts"):
        build_macros(primary(), payload, repeated(), adaptive())
