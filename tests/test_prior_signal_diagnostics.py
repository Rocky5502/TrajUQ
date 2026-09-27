from __future__ import annotations

import hashlib
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_binding(commit: str, cal: Path, test: Path) -> dict:
    return {
        "protocol": "trajuq_analysis_source_binding_v1",
        "git_commit": commit,
        "git_clean": True,
        "source_sha256": {
            "calibration": sha256(cal),
            "test": sha256(test),
        },
    }


def make_spec(
    *,
    commit: str,
    cal: Path,
    test: Path,
    trajuq: Path,
    prior: Path,
) -> dict:
    return {
        "cell": "llada:gsm8k",
        "experiment_git_commit": commit,
        "inputs": {
            "calibration": {
                "path": str(cal),
                "sha256": sha256(cal),
            },
            "test": {
                "path": str(test),
                "sha256": sha256(test),
            },
        },
        "outputs": {
            "predictions": str(trajuq),
            "prior_signal_predictions": str(prior),
        },
    }


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
    commit = "a" * 40
    cal = tmp_path / "calibration.scored.jsonl"
    test = tmp_path / "test.scored.jsonl"
    trajuq = tmp_path / "predictions.jsonl"
    prior = tmp_path / "prior.jsonl"
    cal.write_text(
        '{"example_id":"c","correct":true}\n',
        encoding="utf-8",
    )
    test.write_text(
        '{"example_id":"a","correct":true}\n'
        '{"example_id":"b","correct":false}\n',
        encoding="utf-8",
    )
    binding = make_binding(commit, cal, test)

    trajuq.write_text(
        json.dumps(
            {
                "example_id": "a",
                "y_error": 0,
                "p_error_trajuq": 0.2,
                "analysis_provenance": binding,
            }
        )
        + "\n"
        + json.dumps(
            {
                "example_id": "b",
                "y_error": 1,
                "p_error_trajuq": 0.8,
                "analysis_provenance": binding,
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
                "correct": True,
                "p_error_trajectory_entropy": 0.3,
                "p_error_top1_revision_rate": 0.4,
                "analysis_provenance": binding,
            }
        )
        + "\n"
        + json.dumps(
            {
                "example_id": "b",
                "y_error": 1,
                "correct": False,
                "p_error_trajectory_entropy": 0.7,
                "p_error_top1_revision_rate": 0.6,
                "analysis_provenance": binding,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    spec = make_spec(
        commit=commit,
        cal=cal,
        test=test,
        trajuq=trajuq,
        prior=prior,
    )
    y, trajuq_score, signals = load_cell(spec)

    assert np.array_equal(y, np.asarray([0, 1]))
    assert np.allclose(trajuq_score, [0.2, 0.8])
    assert set(signals) == {
        "trajectory_entropy",
        "top1_revision_rate",
    }


def test_prior_signal_macro_loader_rejects_reordered_ids(tmp_path):
    commit = "a" * 40
    cal = tmp_path / "calibration.scored.jsonl"
    test = tmp_path / "test.scored.jsonl"
    trajuq = tmp_path / "predictions.jsonl"
    prior = tmp_path / "prior.jsonl"
    cal.write_text(
        '{"example_id":"c","correct":true}\n',
        encoding="utf-8",
    )
    test.write_text(
        '{"example_id":"a","correct":true}\n'
        '{"example_id":"b","correct":false}\n',
        encoding="utf-8",
    )
    binding = make_binding(commit, cal, test)

    trajuq.write_text(
        json.dumps(
            {
                "example_id": "a",
                "y_error": 0,
                "p_error_trajuq": 0.2,
                "analysis_provenance": binding,
            }
        )
        + "\n"
        + json.dumps(
            {
                "example_id": "b",
                "y_error": 1,
                "p_error_trajuq": 0.8,
                "analysis_provenance": binding,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prior.write_text(
        json.dumps(
            {
                "example_id": "b",
                "y_error": 1,
                "correct": False,
                "p_error_trajectory_entropy": 0.7,
                "p_error_top1_revision_rate": 0.6,
                "analysis_provenance": binding,
            }
        )
        + "\n"
        + json.dumps(
            {
                "example_id": "a",
                "y_error": 0,
                "correct": True,
                "p_error_trajectory_entropy": 0.3,
                "p_error_top1_revision_rate": 0.4,
                "analysis_provenance": binding,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    spec = make_spec(
        commit=commit,
        cal=cal,
        test=test,
        trajuq=trajuq,
        prior=prior,
    )
    with pytest.raises(SystemExit):
        load_cell(spec)
