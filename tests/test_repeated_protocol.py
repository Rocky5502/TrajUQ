import hashlib
import json

import pytest

from scripts.run_repeated_samples import verified_raw_artifact
from trajuq.repeated_sampling import vote_uncertainty


def test_five_seed_protocol_has_expected_size():
    seeds = [2027, 2028, 2029, 2030, 2031]
    assert len(seeds) == 5
    assert len(seeds) == len(set(seeds))


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

    assert verified_raw_artifact(raw, "frozen-commit")
    assert not verified_raw_artifact(raw, "other-commit")

    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["git_clean"] = False
    audit.write_text(json.dumps(payload), encoding="utf-8")
    assert not verified_raw_artifact(raw, "frozen-commit")
