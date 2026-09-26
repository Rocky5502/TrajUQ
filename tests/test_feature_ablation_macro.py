import json

import numpy as np
import pytest

from scripts.aggregate_feature_ablation import load_cell
from trajuq.features import FROZEN_FEATURE_GROUPS


def test_load_ablation_cell_requires_exact_frozen_groups(tmp_path):
    target = tmp_path / "test.scored.jsonl"
    predictions = tmp_path / "feature_ablation_predictions.jsonl"

    target.write_text(
        '{"example_id":"a","correct":true}\n'
        '{"example_id":"b","correct":false}\n',
        encoding="utf-8",
    )
    rows = []
    for item_id, correct, y_error in (
        ("a", True, 0),
        ("b", False, 1),
    ):
        rows.append(
            {
                "example_id": item_id,
                "correct": correct,
                "y_error": y_error,
                "p_error_by_group": {
                    name: 0.25 if correct else 0.75
                    for name in FROZEN_FEATURE_GROUPS
                },
            }
        )
    predictions.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    spec = {
        "cell": "llada:gsm8k",
        "outputs": {
            "feature_ablation_predictions": str(predictions),
        },
        "inputs": {
            "test": {
                "path": str(target),
            }
        },
    }

    ids, y_error, scores = load_cell(spec)
    assert ids == ["a", "b"]
    assert np.array_equal(y_error, np.asarray([0, 1]))
    assert set(scores) == set(FROZEN_FEATURE_GROUPS)
    assert all(len(values) == 2 for values in scores.values())


def test_load_ablation_cell_rejects_group_drift(tmp_path):
    target = tmp_path / "test.scored.jsonl"
    predictions = tmp_path / "feature_ablation_predictions.jsonl"

    target.write_text(
        '{"example_id":"a","correct":false}\n',
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            {
                "example_id": "a",
                "correct": False,
                "y_error": 1,
                "p_error_by_group": {"decision_only": 0.5},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = {
        "cell": "llada:gsm8k",
        "outputs": {
            "feature_ablation_predictions": str(predictions),
        },
        "inputs": {
            "test": {
                "path": str(target),
            }
        },
    }

    with pytest.raises(SystemExit, match="group set drifted"):
        load_cell(spec)
