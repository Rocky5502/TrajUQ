import hashlib
import json

import pytest

import scripts.score_outputs as score_outputs
from scripts.score_outputs import (
    reference_index,
    validate_record_source,
    verify_raw_audit,
)
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


def test_verify_raw_audit_requires_clean_same_commit_hash_binding(
    tmp_path,
    monkeypatch,
):
    raw = tmp_path / "run.raw.jsonl"
    raw.write_text('{"example_id":"x"}\n', encoding="utf-8")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    audit = raw.with_suffix(raw.suffix + ".audit.json")
    audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "stage": "raw",
                "git_clean": True,
                "git_commit": "frozen-commit",
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        score_outputs,
        "git_head",
        lambda: "frozen-commit",
    )

    path, payload = verify_raw_audit(raw)
    assert path == audit
    assert payload["status"] == "PASS"

    raw.write_text('{"example_id":"tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash-bound PASS"):
        verify_raw_audit(raw)


def test_verify_raw_audit_rejects_other_commit(tmp_path, monkeypatch):
    raw = tmp_path / "run.raw.jsonl"
    raw.write_text('{"example_id":"x"}\n', encoding="utf-8")
    audit = raw.with_suffix(raw.suffix + ".audit.json")
    audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "stage": "raw",
                "git_clean": True,
                "git_commit": "other-commit",
                "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        score_outputs,
        "git_head",
        lambda: "frozen-commit",
    )

    with pytest.raises(ValueError, match="same-commit"):
        verify_raw_audit(raw)
