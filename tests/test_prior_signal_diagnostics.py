from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.aggregate_prior_signal_diagnostics import load_cell
from scripts.evaluate_prior_signal_diagnostics import (
    SIGNALS,
    feature_values,
    labels,
)


def test_prior_signal_feature_views_are_frozen():
    assert SIGNALS == {
        "trajectory_entropy": {
            "feature": "entropy_mean",
            "relation_to_prior_work": SIGNALS[
                "trajectory_entropy"
            ]["relation_to_prior_work"],
        },
        "top1_revision_rate": {
            "feature": "revision_rate",
            "relation_to_prior_work": SIGNALS[
                "top1_revision_rate"
            ]["relation_to_prior_work"],
        },
    }


def test_feature_values_and_labels_use_scored_frozen_rows():
    rows = [
        {
            "example_id": "a",
            "correct": True,
            "generation": {
                "features": {
                    "entropy_mean": 0.2,
                    "revision_rate": 0.1,
                }
            },
        },
        {
            "example_id": "b",
            "correct": False,
            "generation": {
                "features": {
                    "entropy_mean": 0.8,
                    "revision_rate": 0.6,
                }
            },
        },
    ]

    assert np.array_equal(labels(rows), np.asarray([0, 1]))
    assert np.allclose(
        feature_values(rows, "entropy_mean"),
        np.asarray([[0.2], [0.8]]),
    )


def test_prior_signal_macro_loader_requires_exact_ids_and_labels(tmp_path):
    trajuq = tmp_path / "predictions.jsonl"
    prior = tmp_path / "prior.jsonl"

    trajuq.write_text(
        json.dumps(
            {
                "example_id": "a",
                "y_error": 0,
                "p_error_trajuq": 0.2,
            }
        )
        + "\n"
        + json.dumps(
            {
                "example_id": "b",
                "y_error": 1,
                "p_error_trajuq": 0.8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prior.write_text(
        json.dumps(
            {
                "example_id": "a",
                "y_error": 0,
                "p_error_trajectory_entropy": 0.3,
                "p_error_top1_revision_rate": 0.4,
            }
        )
        + "\n"
        + json.dumps(
            {
                "example_id": "b",
                "y_error": 1,
                "p_error_trajectory_entropy": 0.7,
                "p_error_top1_revision_rate": 0.6,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    spec = {
        "cell": "llada:gsm8k",
        "outputs": {
            "predictions": str(trajuq),
            "prior_signal_predictions": str(prior),
        },
    }
    y, trajuq_score, signals = load_cell(spec)

    assert np.array_equal(y, np.asarray([0, 1]))
    assert np.allclose(trajuq_score, [0.2, 0.8])
    assert set(signals) == {
        "trajectory_entropy",
        "top1_revision_rate",
    }


def test_prior_signal_macro_loader_rejects_reordered_ids(tmp_path):
    trajuq = tmp_path / "predictions.jsonl"
    prior = tmp_path / "prior.jsonl"

    trajuq.write_text(
        '{"example_id":"a","y_error":0,"p_error_trajuq":0.2}\n'
        '{"example_id":"b","y_error":1,"p_error_trajuq":0.8}\n',
        encoding="utf-8",
    )
    prior.write_text(
        '{"example_id":"b","y_error":1,'
        '"p_error_trajectory_entropy":0.7,'
        '"p_error_top1_revision_rate":0.6}\n'
        '{"example_id":"a","y_error":0,'
        '"p_error_trajectory_entropy":0.3,'
        '"p_error_top1_revision_rate":0.4}\n',
        encoding="utf-8",
    )

    spec = {
        "cell": "llada:gsm8k",
        "outputs": {
            "predictions": str(trajuq),
            "prior_signal_predictions": str(prior),
        },
    }
    with pytest.raises(SystemExit, match="do not exactly match"):
        load_cell(spec)
