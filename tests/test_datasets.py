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
