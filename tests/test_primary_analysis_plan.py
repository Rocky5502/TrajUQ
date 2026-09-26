import hashlib
import json
from scripts.plan_primary_analysis import (
    audit_passes,
    exact_id_coverage,
    main as analysis_main,
)


def test_primary_analysis_planner_imports():
    assert callable(analysis_main)


def test_primary_analysis_requires_frozen_scored_audit(tmp_path):
    scored = tmp_path / "cell.scored.jsonl"
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

    assert audit_passes(scored, "frozen-commit")
    assert not audit_passes(scored, "other-commit")

    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["git_clean"] = False
    audit.write_text(json.dumps(payload), encoding="utf-8")
    assert not audit_passes(scored, "frozen-commit")


def test_primary_analysis_requires_exact_frozen_input_coverage(tmp_path):
    source = tmp_path / "calibration.jsonl"
    scored = tmp_path / "calibration.scored.jsonl"
    source.write_text(
        '{"id":"a"}\n{"id":"b"}\n',
        encoding="utf-8",
    )
    scored.write_text(
        '{"example_id":"a"}\n',
        encoding="utf-8",
    )

    assert not exact_id_coverage(source, scored)

    scored.write_text(
        '{"example_id":"a"}\n{"example_id":"b"}\n',
        encoding="utf-8",
    )
    assert exact_id_coverage(source, scored)


def test_primary_analysis_rejects_reordered_or_duplicate_ids(tmp_path):
    source = tmp_path / "test.jsonl"
    scored = tmp_path / "test.scored.jsonl"
    source.write_text(
        '{"id":"a"}\n{"id":"b"}\n',
        encoding="utf-8",
    )

    scored.write_text(
        '{"example_id":"b"}\n{"example_id":"a"}\n',
        encoding="utf-8",
    )
    assert not exact_id_coverage(source, scored)

    scored.write_text(
        '{"example_id":"a"}\n{"example_id":"a"}\n',
        encoding="utf-8",
    )
    assert not exact_id_coverage(source, scored)
