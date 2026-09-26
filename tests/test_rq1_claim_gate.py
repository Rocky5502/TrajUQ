from scripts.evaluate_rq1_claim_gate import evaluate_gate


def config():
    return {
        "statistics": {
            "primary_endpoint": "hierarchical_macro_auroc_improvement",
            "rq1_claim_gate": {
                "macro_auroc_ci_low_gt_zero": True,
                "min_positive_model_families": 2,
            },
        }
    }


def macro(ci_low: float, positive_families: int, complete: bool = True):
    return {
        "cell_set_complete": complete,
        "macro": {
            "auroc": {
                "improvement_macro": 0.04,
                "improvement_ci_low": ci_low,
                "improvement_ci_high": 0.08,
                "positive_model_families": positive_families,
                "n_model_families": 3,
            }
        },
    }


def test_rq1_gate_requires_ci_and_cross_family_consistency():
    result = evaluate_gate(macro(0.01, 2), config())
    assert result["claim_supported"] is True
    assert result["decision"] == "SUPPORT_POSITIVE_RQ1_CLAIM"


def test_rq1_gate_preserves_negative_result():
    result = evaluate_gate(macro(-0.01, 3), config())
    assert result["claim_supported"] is False
    assert result["decision"] == "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM"


def test_rq1_gate_rejects_partial_cell_set():
    result = evaluate_gate(macro(0.01, 3, complete=False), config())
    assert result["claim_supported"] is False
