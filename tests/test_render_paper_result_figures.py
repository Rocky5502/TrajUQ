import numpy as np
import pytest

from scripts.render_paper_result_figures import (
    auroc_improvement_matrix,
    require_complete_adaptive,
    require_complete_primary,
)


def primary_payload():
    return {
        "protocol": "hierarchical_within_cell_macro_bootstrap_v1",
        "cell_set_complete": True,
        "expected_cell_count": 2,
        "observed_cell_count": 2,
        "per_cell": {
            "llada_a": {
                "model_family": "llada",
                "dataset": "a",
                "metrics": {"auroc": {"improvement": 0.1}},
            },
            "llada_b": {
                "model_family": "llada",
                "dataset": "b",
                "metrics": {"auroc": {"improvement": -0.2}},
            },
        },
    }


def test_auroc_matrix_respects_frozen_order():
    matrix = auroc_improvement_matrix(
        primary_payload(),
        ["llada"],
        ["b", "a"],
    )
    assert matrix.shape == (1, 2)
    assert np.allclose(matrix, [-0.2, 0.1])


def test_primary_figure_gate_rejects_incomplete_matrix():
    payload = primary_payload()
    payload["cell_set_complete"] = False
    with pytest.raises(ValueError):
        require_complete_primary(payload, 2)


def adaptive_payload(
    *,
    key: str = "cap_050",
    fraction: float = 0.50,
    primary: bool = True,
) -> dict:
    return {
        "protocol": "trajuq_adaptive_macro_v1",
        "execution_semantics": "selective_second_pass_rerun",
        "fraction_key": key,
        "cell_set_complete": True,
        "expected_cells": 18,
        "observed_cells": 18,
        "is_primary_fraction": primary,
        "max_refine_fraction": fraction,
    }


def test_adaptive_figure_gate_requires_complete_cells():
    payload = adaptive_payload()
    require_complete_adaptive(
        payload,
        18,
        fraction_key="cap_050",
        expected_fraction=0.50,
        is_primary=True,
    )
    payload["observed_cells"] = 17
    with pytest.raises(ValueError):
        require_complete_adaptive(
            payload,
            18,
            fraction_key="cap_050",
            expected_fraction=0.50,
            is_primary=True,
        )


def test_adaptive_figure_gate_rejects_fraction_key_drift():
    payload = adaptive_payload(key="cap_025", fraction=0.25, primary=False)
    with pytest.raises(ValueError, match="fraction key drifted"):
        require_complete_adaptive(
            payload,
            18,
            fraction_key="cap_050",
            expected_fraction=0.50,
            is_primary=True,
        )
