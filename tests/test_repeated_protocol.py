import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.run_repeated_samples import repeated_seed_contract, verified_raw_artifact
from trajuq.repeated_sampling import vote_uncertainty


def test_five_seed_protocol_has_expected_size():
    seeds = [2027, 2028, 2029, 2030, 2031]
    assert len(seeds) == 5
    assert len(seeds) == len(set(seeds))


def test_repeated_seed_reuse_rejects_wrong_sampler_settings(tmp_path):
    raw = tmp_path / "seed-2027.raw.jsonl"
    args = SimpleNamespace(
        adapter="llada",
        model="example/llada",
        revision="model-sha",
        temperature=0.2,
        top_p=1.0,
        steps=128,
        max_new_tokens=128,
        block_length=128,
        llada_cfg_scale=0.0,
        llada_remasking="low_confidence",
        dream_alg="entropy",
        dream_alg_temp=0.1,
        sdar_denoising_steps=4,
        sdar_top_k=0,
        sdar_remasking_strategy="low_confidence_dynamic",
        sdar_confidence_threshold=0.85,
    )
    row = {
        "example_id": "x",
        "adapter": "llada",
        "model": "example/llada",
        "seed": 2027,
        "generation": {
            "metadata": {
                "model_revision": "model-sha",
                "temperature": 0.9,
                "steps_requested": 128,
                "gen_length": 128,
                "block_length": 128,
                "cfg_scale": 0.0,
                "remasking": "low_confidence",
            }
        },
    }
    raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    raw.with_suffix(raw.suffix + ".audit.json").write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": "frozen-commit",
                "git_clean": True,
                "stage": "raw",
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    assert not verified_raw_artifact(
        raw,
        "frozen-commit",
        ["x"],
        repeated_seed_contract(args, 2027),
    )


def test_repeated_seed_reuse_rejects_other_source_commit(tmp_path):
    raw = tmp_path / "seed-2027.raw.jsonl"
    args = SimpleNamespace(
        adapter="llada",
        model="example/llada",
        revision="model-sha",
        temperature=0.2,
        top_p=1.0,
        steps=128,
        max_new_tokens=128,
        block_length=128,
        llada_cfg_scale=0.0,
        llada_remasking="low_confidence",
        dream_alg="entropy",
        dream_alg_temp=0.1,
        sdar_denoising_steps=4,
        sdar_top_k=0,
        sdar_remasking_strategy="low_confidence_dynamic",
        sdar_confidence_threshold=0.85,
    )
    row = {
        "example_id": "x",
        "adapter": "llada",
        "model": "example/llada",
        "seed": 2027,
        "generation": {
            "metadata": {
                "model_revision": "model-sha",
                "temperature": 0.2,
                "steps_requested": 128,
                "gen_length": 128,
                "block_length": 128,
                "cfg_scale": 0.0,
                "remasking": "low_confidence",
            }
        },
        "metadata": {
            "source_git_commit": "old-commit",
            "source_git_clean": True,
        },
    }
    raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    raw.with_suffix(raw.suffix + ".audit.json").write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": "frozen-commit",
                "git_clean": True,
                "stage": "raw",
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    assert not verified_raw_artifact(
        raw,
        "frozen-commit",
        ["x"],
        repeated_seed_contract(args, 2027, "frozen-commit"),
    )


def test_vote_uncertainty_requires_multiple_samples():
    with pytest.raises(ValueError):
        vote_uncertainty(["A"])


def test_repeated_seed_reuse_requires_frozen_clean_audit(tmp_path):
    raw = tmp_path / "seed-2027.raw.jsonl"
    raw.write_text('{"example_id":"x"}\n', encoding="utf-8")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    audit = raw.with_suffix(raw.suffix + ".audit.json")
    audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": "frozen-commit",
                "git_clean": True,
                "stage": "raw",
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    assert verified_raw_artifact(raw, "frozen-commit", ["x"])
    assert not verified_raw_artifact(raw, "other-commit", ["x"])
    assert not verified_raw_artifact(raw, "frozen-commit", ["x", "y"])

    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["git_clean"] = False
    audit.write_text(json.dumps(payload), encoding="utf-8")
    assert not verified_raw_artifact(raw, "frozen-commit", ["x"])
