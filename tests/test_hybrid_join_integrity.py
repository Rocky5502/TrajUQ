import pytest

from scripts.join_hybrid_paths import assert_pre_outcome, checkpoint_identity


def test_checkpoint_identity_reads_generation_metadata():
    row = {
        "model": "checkpoint",
        "generation": {"metadata": {"model_revision": "sha"}},
    }
    assert checkpoint_identity(row) == ("checkpoint", "sha")


def test_join_rejects_scored_path_artifact():
    with pytest.raises(SystemExit):
        assert_pre_outcome(
            {"correct": True, "reference": None, "metadata": {}},
            "ar.jsonl",
            "x",
        )


def test_join_rejects_outcome_like_metadata():
    with pytest.raises(SystemExit):
        assert_pre_outcome(
            {"correct": None, "reference": None, "metadata": {"y_error": 1}},
            "diff.jsonl",
            "x",
        )
