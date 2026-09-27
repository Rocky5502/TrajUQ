import hashlib
import json
from pathlib import Path

from scripts.plan_primary_analysis import (
    audit_passes,
    exact_id_coverage,
    main as analysis_main,
    scored_source_chain_passes,
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


def test_scored_source_chain_requires_raw_audit_and_metadata_bindings(
    tmp_path,
):
    source = tmp_path / "input.jsonl"
    raw = tmp_path / "run.raw.jsonl"
    raw_audit = raw.with_suffix(raw.suffix + ".audit.json")
    scored = tmp_path / "run.scored.jsonl"
    source.write_text('{"id":"x"}\n', encoding="utf-8")
    raw.write_text('{"example_id":"x"}\n', encoding="utf-8")
    raw_audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": "frozen-commit",
                "git_clean": True,
                "stage": "raw",
                "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "scorer": "gsm8k_numeric_exact",
        "raw_run_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "raw_audit_sha256": hashlib.sha256(raw_audit.read_bytes()).hexdigest(),
        "raw_audit_git_commit": "frozen-commit",
        "scoring_input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    scored.write_text(
        json.dumps({"example_id": "x", "metadata": metadata}) + "\n",
        encoding="utf-8",
    )

    assert scored_source_chain_passes(
        source,
        raw,
        scored,
        "frozen-commit",
    )

    payload = json.loads(scored.read_text(encoding="utf-8"))
    payload["metadata"]["raw_run_sha256"] = "tampered"
    scored.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert not scored_source_chain_passes(
        source,
        raw,
        scored,
        "frozen-commit",
    )


def test_primary_macro_command_is_bound_to_analysis_plan():
    text = Path("scripts/plan_primary_analysis.py").read_text(encoding="utf-8")

    assert '"--analysis-plan"' in text
    assert "args.output" in text


def test_primary_macro_aggregator_requires_frozen_test_coverage():
    text = Path("scripts/aggregate_cells.py").read_text(encoding="utf-8")

    assert "validate_plan_provenance(analysis_plan)" in text
    assert "predictions do not exactly cover the frozen" in text
    assert "scored test target" in text


def test_primary_analysis_plan_wires_ablation_predictions_and_macro():
    text = Path("scripts/plan_primary_analysis.py").read_text(encoding="utf-8")

    assert '"--predictions-output"' in text
    assert "feature_ablation_predictions.jsonl" in text
    assert "aggregate_feature_ablation.py" in text
    assert "macro_feature_ablation.json" in text
    assert '"feature_ablation_macro_argv"' in text


def test_primary_analysis_plan_wires_prior_signal_diagnostics():
    text = Path("scripts/plan_primary_analysis.py").read_text(
        encoding="utf-8"
    )

    assert "evaluate_prior_signal_diagnostics.py" in text
    assert "prior_signal_predictions.jsonl" in text
    assert "aggregate_prior_signal_diagnostics.py" in text
    assert "macro_prior_signal_diagnostics.json" in text
    assert '"prior_signal_macro_argv"' in text


def test_primary_analysis_planner_revalidates_primary_plan_provenance():
    text = Path("scripts/plan_primary_analysis.py").read_text(encoding="utf-8")

    assert "validate_primary_plan_provenance" in text
    assert "primary plan provenance: " in text
