import json

import pytest

from scripts.aggregate_repeated_baseline import load_cell
from trajuq.repeated_sampling import equivalence_cluster_labels, vote_uncertainty


def test_vote_uncertainty_zero_when_all_samples_agree():
    result = vote_uncertainty(["A", "A", "A", "A", "A"])
    assert result["sample_disagreement"] == 0.0
    assert result["sample_vote_entropy"] == 0.0
    assert result["sample_unique_fraction"] == 0.2


def test_vote_uncertainty_increases_with_disagreement():
    agree = vote_uncertainty(["A", "A", "A", "A"])
    split = vote_uncertainty(["A", "A", "B", "B"])
    all_different = vote_uncertainty(["A", "B", "C", "D"])
    assert agree["sample_disagreement"] < split["sample_disagreement"]
    assert split["sample_disagreement"] < all_different["sample_disagreement"]
    assert agree["sample_vote_entropy"] < split["sample_vote_entropy"]
    assert split["sample_vote_entropy"] <= all_different["sample_vote_entropy"]



def test_equivalence_cluster_labels_are_deterministic():
    values = ["1/2", "0.5", "3/4", "0.500"]
    equivalent_pairs = {
        frozenset(("1/2", "0.5")),
        frozenset(("1/2", "0.500")),
        frozenset(("0.5", "0.500")),
    }

    def equivalent(left, right):
        return left == right or frozenset((left, right)) in equivalent_pairs

    labels = equivalence_cluster_labels(values, equivalent)
    assert labels[0] == labels[1] == labels[3]
    assert labels[2] != labels[0]


def test_repeated_macro_rejects_partial_frozen_test_coverage(tmp_path):
    target = tmp_path / "test.scored.jsonl"
    traj = tmp_path / "trajuq.jsonl"
    repeated = tmp_path / "repeated.jsonl"
    target.write_text(
        '{"example_id":"a"}\n{"example_id":"b"}\n',
        encoding="utf-8",
    )
    traj.write_text('{"example_id":"a"}\n', encoding="utf-8")
    repeated.write_text('{"example_id":"a"}\n', encoding="utf-8")
    spec = {
        "cell": "llada:gsm8k",
        "adapter": "llada",
        "dataset": "gsm8k",
        "trajuq_predictions": str(traj),
        "repeated_predictions": str(repeated),
        "provenance": {
            "test": {"scored_target": str(target)}
        },
    }

    with pytest.raises(SystemExit, match="exactly cover"):
        load_cell(spec)
