import numpy as np

from trajuq.aggregation import PredictionCell, hierarchical_macro_bootstrap


def test_macro_reports_model_family_consistency():
    y = np.array([0, 0, 0, 1, 1, 1])
    good = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    bad = np.array([0.6, 0.2, 0.7, 0.1, 0.8, 0.4])
    cells = [
        PredictionCell("llada:a", y, good, bad, model_family="llada", dataset="a"),
        PredictionCell("dream:a", y, good, bad, model_family="dream", dataset="a"),
        PredictionCell("sdar:a", y, good, bad, model_family="sdar", dataset="a"),
    ]
    result = hierarchical_macro_bootstrap(
        cells,
        metric="auroc",
        n_boot=100,
        seed=5,
    )
    assert result["n_model_families"] == 3
    assert result["positive_model_families"] == 3
    assert set(result["model_family_macro_improvement"]) == {"llada", "dream", "sdar"}
