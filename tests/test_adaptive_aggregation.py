import numpy as np

from trajuq.adaptive_aggregation import (
    AdaptiveCell,
    cell_statistics,
    hierarchical_macro_adaptive,
)


def make_cell(label: str, adapter: str = "llada") -> AdaptiveCell:
    return AdaptiveCell(
        label=label,
        adapter=adapter,
        dataset="gsm8k",
        base_correct=np.array([1, 1, 0, 0], dtype=float),
        refined_correct=np.array([1, 1, 1, 1], dtype=float),
        selected=np.array([0, 0, 1, 1], dtype=bool),
        base_compute=32.0,
        refined_compute=128.0,
    )


def test_cell_statistics_second_pass_cost_and_accuracy():
    stats = cell_statistics(make_cell("a"))
    assert stats["adaptive_accuracy"] == 1.0
    assert stats["selected_fraction"] == 0.5
    assert stats["nominal_compute"] == 96.0
    assert stats["nominal_cost_ratio_to_base"] == 3.0
    assert stats["delta_vs_random"] > 0


def test_hierarchical_adaptive_bootstrap_is_reproducible():
    cells = [make_cell("a"), make_cell("b", adapter="dream")]
    a = hierarchical_macro_adaptive(cells, n_boot=100, seed=11)
    b = hierarchical_macro_adaptive(cells, n_boot=100, seed=11)
    assert a == b
    assert a["n_cells"] == 2
    assert a["macro"]["delta_vs_base"] > 0
    assert a["intervals"]["adaptive_accuracy"]["point"] == 1.0
