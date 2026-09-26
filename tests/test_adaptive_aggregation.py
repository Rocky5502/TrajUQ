import json

import numpy as np
import pytest

from scripts.aggregate_adaptive import load_prediction_cell

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


def test_adaptive_macro_rejects_partial_frozen_test_coverage(tmp_path):
    target = tmp_path / "refined.scored.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    result = tmp_path / "result.json"
    target.write_text(
        '{"example_id":"a"}\n{"example_id":"b"}\n',
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            {
                "example_id": "a",
                "threshold": 0.5,
                "max_refine_fraction": 0.5,
                "execution_semantics": "selective_second_pass_rerun",
                "selected": False,
                "base_compute": 32,
                "refined_compute": 128,
                "base_correct": True,
                "refined_correct": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result.write_text(
        json.dumps(
            {
                "protocol": "trajuq_adaptive_evaluation_v2",
                "execution_semantics": "selective_second_pass_rerun",
                "n_test": 1,
                "max_refine_fraction": 0.5,
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "cell": "llada:gsm8k",
        "adapter": "llada",
        "dataset": "gsm8k",
        "provenance": {
            "test": {"refined_scored": str(target)}
        },
        "evaluations": {
            "cap_050": {
                "predictions": str(predictions),
                "result": str(result),
                "max_refine_fraction": 0.5,
            }
        },
    }

    with pytest.raises(SystemExit, match="exactly cover"):
        load_prediction_cell(spec, "cap_050")
