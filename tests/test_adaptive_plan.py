import hashlib
import json

from scripts.plan_adaptive_runs import (
    fraction_key,
    replace_arg,
    scored_audit_passes,
)


def test_replace_arg_updates_existing_value():
    argv = ["python", "x.py", "--steps", "128", "--output", "old"]
    out = replace_arg(argv, "--steps", "32")
    assert out == ["python", "x.py", "--steps", "32", "--output", "old"]
    assert argv[3] == "128"


def test_replace_arg_appends_missing_flag():
    assert replace_arg(["python", "x.py"], "--steps", "32")[-2:] == ["--steps", "32"]


def test_fraction_key_is_stable():
    assert fraction_key(0.25) == "cap_025"
    assert fraction_key(0.50) == "cap_050"
    assert fraction_key(0.75) == "cap_075"


def test_adaptive_plan_requires_frozen_scored_audit(tmp_path):
    scored = tmp_path / "refined.scored.jsonl"
    scored.write_text('{"example_id":"x"}\n', encoding="utf-8")
    digest = hashlib.sha256(scored.read_bytes()).hexdigest()
    audit = scored.with_suffix(scored.suffix + ".audit.json")
    audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": "frozen-commit",
                "git_clean": True,
                "stage": "scored",
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    assert scored_audit_passes(scored, "frozen-commit")
    assert not scored_audit_passes(scored, "other-commit")
