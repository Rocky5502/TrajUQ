import hashlib
import json
from pathlib import Path

from scripts.plan_repeated_sampling import (
    arg_value,
    jsonl_example_ids,
    repeated_generation_argv,
    scored_audit_passes,
)


def llada_cell():
    return {
        "adapter": "llada",
        "model_id": "m",
        "model_revision": "sha",
        "calibration": {
            "input": "cal.jsonl",
            "generate_argv": [
                "python",
                "scripts/run_generation.py",
                "--steps",
                "128",
                "--max-new-tokens",
                "128",
                "--block-length",
                "128",
                "--llada-cfg-scale",
                "0.0",
                "--llada-remasking",
                "low_confidence",
            ],
        },
    }


def test_arg_value_reads_frozen_flag():
    assert arg_value(["x", "--steps", "128"], "--steps") == "128"


def test_repeated_plan_uses_all_frozen_seeds_and_nonzero_temperature():
    cmd = repeated_generation_argv(
        llada_cell(),
        "calibration",
        "out",
        [2027, 2028, 2029, 2030, 2031],
        0.2,
        1.0,
    )
    assert cmd[cmd.index("--temperature") + 1] == "0.2"
    seed_idx = cmd.index("--seeds")
    assert cmd[seed_idx + 1 : seed_idx + 6] == [
        "2027",
        "2028",
        "2029",
        "2030",
        "2031",
    ]
    assert cmd[cmd.index("--steps") + 1] == "128"


def test_repeated_plan_requires_frozen_scored_audit(tmp_path):
    scored = tmp_path / "target.scored.jsonl"
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


def test_repeated_plan_prediction_ids_are_exact_and_unique(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        '{"example_id":"a"}\n{"example_id":"b"}\n',
        encoding="utf-8",
    )
    assert jsonl_example_ids(path) == ["a", "b"]

    path.write_text(
        '{"example_id":"a"}\n{"example_id":"a"}\n',
        encoding="utf-8",
    )
    try:
        jsonl_example_ids(path)
    except ValueError as exc:
        assert "duplicate example IDs" in str(exc)
    else:
        raise AssertionError("duplicate prediction IDs must be rejected")


def test_repeated_planner_checks_prediction_coverage_contract():
    text = Path("scripts/plan_repeated_sampling.py").read_text(encoding="utf-8")
    assert "predictions do not exactly" in text
    assert "cover the frozen scored test target" in text
