from trajuq.datasets import namespace_record_id, normalize_row, split_bucket


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
