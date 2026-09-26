import numpy as np

from trajuq.aggregation import (
    PredictionCell,
    hierarchical_macro_bootstrap,
    oriented_improvement,
)


def test_oriented_improvement_positive_for_better_candidate():
    y = np.array([0, 0, 0, 1, 1, 1])
    good = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    bad = np.array([0.6, 0.2, 0.7, 0.1, 0.8, 0.4])
    assert oriented_improvement(y, good, bad, "auroc") > 0
    assert oriented_improvement(y, good, bad, "aurc") > 0


def test_hierarchical_macro_bootstrap_equal_weights_cells():
    y = np.array([0, 0, 0, 1, 1, 1])
    good = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    bad = np.array([0.6, 0.2, 0.7, 0.1, 0.8, 0.4])
    cells = [
        PredictionCell("a", y, good, bad),
        PredictionCell("b", y, good, bad),
    ]
    result = hierarchical_macro_bootstrap(
        cells,
        metric="auroc",
        n_boot=100,
        seed=3,
    )
    assert result["n_cells"] == 2
    assert result["positive_cells"] == 2
    assert result["improvement_macro"] > 0
    assert result["improvement_ci_low"] > 0
