from trajuq.datasets import (
    SPLIT_SELECTION_VERSION,
    deterministic_cap_rows,
    namespace_record_id,
    normalize_row,
    split_bucket,
)


def test_split_bucket_deterministic():
    a = split_bucket("gsm8k", "42", "2027")
    b = split_bucket("gsm8k", "42", "2027")
    c = split_bucket("gsm8k", "43", "2027")
    assert a == b
    assert a != c
    assert 0.0 <= a < 1.0


def test_arc_normalization():
    row = {
        "id": "x",
        "question": "Q?",
        "choices": {"text": ["one", "two"], "label": ["A", "B"]},
        "answerKey": "B",
    }
    rec = normalize_row("arc_challenge", row, 0)
    assert rec["id"] == "x"
    assert rec["reference"] == "B"
    assert "A. one" in rec["prompt"]



def test_namespaced_ids_do_not_collide_across_official_splits():
    train_id = namespace_record_id("gsm8k", "train", "0")
    test_id = namespace_record_id("gsm8k", "test", "0")
    assert train_id == "gsm8k:train:0"
    assert test_id == "gsm8k:test:0"
    assert train_id != test_id


def test_arc_numeric_labels_are_canonicalized_to_letters():
    row = {
        "id": "numeric",
        "question": "Q?",
        "choices": {
            "text": ["one", "two", "three", "four"],
            "label": ["1", "2", "3", "4"],
        },
        "answerKey": "3",
    }
    rec = normalize_row("arc_challenge", row, 0)

    assert rec["reference"] == "C"
    assert "A. one" in rec["prompt"]
    assert "C. three" in rec["prompt"]
    assert "1. one" not in rec["prompt"]


def test_arc_letter_labels_preserve_answer_position():
    row = {
        "id": "letters",
        "question": "Q?",
        "choices": {
            "text": ["one", "two", "three", "four"],
            "label": ["A", "B", "C", "D"],
        },
        "answerKey": "D",
    }
    rec = normalize_row("arc_challenge", row, 0)

    assert rec["reference"] == "D"


def test_deterministic_cap_rows_is_outcome_free_and_stable():
    rows = [
        {"id": f"gsm8k:test:{i}", "reference": f"gold-{i}", "prompt": f"q-{i}"}
        for i in range(20)
    ]
    first = deterministic_cap_rows(
        rows,
        dataset="gsm8k",
        role="test",
        seed="2027",
        max_examples=7,
    )
    changed_outcomes = [
        {**row, "reference": "CHANGED", "prompt": "CHANGED"}
        for row in reversed(rows)
    ]
    second = deterministic_cap_rows(
        changed_outcomes,
        dataset="gsm8k",
        role="test",
        seed="2027",
        max_examples=7,
    )

    assert SPLIT_SELECTION_VERSION == "sha256_id_rank_v1"
    assert len(first) == 7
    assert [row["id"] for row in first] == [row["id"] for row in second]


def test_deterministic_cap_rows_preserves_small_partition_and_rejects_bad_cap():
    rows = [{"id": "x:0"}, {"id": "x:1"}]

    assert deterministic_cap_rows(
        rows,
        dataset="x",
        role="calibration",
        seed="2027",
        max_examples=10,
    ) == rows

    import pytest

    with pytest.raises(ValueError, match="positive integer"):
        deterministic_cap_rows(
            rows,
            dataset="x",
            role="calibration",
            seed="2027",
            max_examples=0,
        )
