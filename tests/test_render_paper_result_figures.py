import numpy as np
import pytest

from scripts.render_paper_result_figures import (
    auroc_improvement_matrix,
    require_complete_adaptive,
    require_complete_primary,
)


def primary_payload():
    return {
        "cell_set_complete": True,
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


def test_adaptive_figure_gate_requires_complete_cells():
    payload = {
        "cell_set_complete": True,
        "observed_cells": 18,
    }
    require_complete_adaptive(payload, 18)
    payload["observed_cells"] = 17
    with pytest.raises(ValueError):
        require_complete_adaptive(payload, 18)
