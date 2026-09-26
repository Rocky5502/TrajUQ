import json

import pytest

from scripts.score_outputs import reference_index, validate_record_source
from trajuq.io import sha256_text


def test_reference_index_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text(
        '{"id":"x","dataset":"gsm8k","prompt":"p","reference":"1"}\n'
        '{"id":"x","dataset":"gsm8k","prompt":"q","reference":"2"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate id"):
        reference_index(path)


def test_scoring_pair_requires_dataset_identity_match():
    source = {
        "id": "x",
        "dataset": "gsm8k",
        "prompt": "question",
        "reference": "#### 1",
    }
    rec = {
        "example_id": "x",
        "dataset": "math500",
        "prompt_sha256": sha256_text("question"),
    }

    with pytest.raises(ValueError, match="differs from frozen scoring input"):
        validate_record_source(rec, source, item_id="x")


def test_scoring_pair_requires_exact_prompt_hash():
    source = {
        "id": "x",
        "dataset": "gsm8k",
        "prompt": "question",
        "reference": "#### 1",
    }
    rec = {
        "example_id": "x",
        "dataset": "gsm8k",
        "prompt_sha256": sha256_text("different question"),
    }

    with pytest.raises(ValueError, match="prompt differs"):
        validate_record_source(rec, source, item_id="x")


def test_scoring_pair_returns_frozen_dataset_name():
    source = {
        "id": "x",
        "dataset": "arc_challenge",
        "prompt": "question",
        "reference": "C",
    }
    rec = {
        "example_id": "x",
        "dataset": "arc_challenge",
        "prompt_sha256": sha256_text("question"),
    }

    assert validate_record_source(rec, source, item_id="x") == "arc_challenge"
