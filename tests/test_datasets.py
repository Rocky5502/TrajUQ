from trajuq.datasets import normalize_row, split_bucket


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
