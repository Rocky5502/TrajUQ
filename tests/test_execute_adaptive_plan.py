import hashlib
import json
from pathlib import Path

import scripts.execute_adaptive_plan as execute_adaptive_plan
from scripts.execute_adaptive_plan import (
    artifact_id_state,
    audit_passes,
    base_partition_state,
    command_needed,
    evaluation_state,
    execution_snapshot,
    macro_state,
    validate_plan_provenance,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_input(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"id": item_id}) + "\n" for item_id in ids),
        encoding="utf-8",
    )


def write_audited(
    path: Path,
    stage: str,
    *,
    ids: list[str] | None = None,
    commit: str = "frozen-commit",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_ids = ids or ["x"]
    path.write_text(
        "".join(
            json.dumps({"example_id": item_id}) + "\n"
            for item_id in resolved_ids
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".audit.json").write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": commit,
                "git_clean": True,
                "stage": stage,
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )


def test_base_partition_requires_both_audited_files(tmp_path):
    source = tmp_path / "input.jsonl"
    raw = tmp_path / "base.raw.jsonl"
    scored = tmp_path / "base.scored.jsonl"
    write_input(source, ["x"])
    spec = {
        "input": str(source),
        "raw_output": str(raw),
        "scored_output": str(scored),
    }

    write_audited(raw, "raw")
    assert not base_partition_state(spec)["complete"]
    write_audited(scored, "scored")
    assert base_partition_state(spec)["complete"]


def test_adaptive_partial_raw_prefix_resumes_but_is_not_complete(tmp_path):
    source = tmp_path / "input.jsonl"
    raw = tmp_path / "base.raw.jsonl"
    scored = tmp_path / "base.scored.jsonl"
    write_input(source, ["a", "b"])
    write_audited(raw, "raw", ids=["a"])
    spec = {
        "input": str(source),
        "raw_output": str(raw),
        "scored_output": str(scored),
    }

    assert artifact_id_state(raw, spec) == "PREFIX"
    assert command_needed("generate", spec, "frozen-commit")
    state = base_partition_state(spec, "frozen-commit")
    assert state["raw_id_state"] == "PREFIX"
    assert not state["complete"]


def test_adaptive_partial_scored_artifact_is_not_overwritten(tmp_path):
    source = tmp_path / "input.jsonl"
    raw = tmp_path / "base.raw.jsonl"
    scored = tmp_path / "base.scored.jsonl"
    write_input(source, ["a", "b"])
    write_audited(raw, "raw", ids=["a", "b"])
    write_audited(scored, "scored", ids=["a"])
    spec = {
        "input": str(source),
        "raw_output": str(raw),
        "scored_output": str(scored),
    }

    try:
        command_needed("score", spec, "frozen-commit")
    except ValueError as exc:
        assert "partial/invalid adaptive scored artifact" in str(exc)
    else:
        raise AssertionError("partial adaptive scored artifact must not be reused")


def test_adaptive_evaluation_rejects_partial_pair(tmp_path):
    result = tmp_path / "result.json"
    predictions = tmp_path / "predictions.jsonl"
    result.write_text("{}", encoding="utf-8")
    spec = {"result": str(result), "predictions": str(predictions)}
    assert evaluation_state(spec) == "PARTIAL"


def test_adaptive_evaluation_rejects_incomplete_prediction_coverage(tmp_path):
    result = tmp_path / "result.json"
    predictions = tmp_path / "predictions.jsonl"
    result.write_text("{}", encoding="utf-8")
    predictions.write_text('{"example_id":"a"}\n', encoding="utf-8")

    spec = {
        "result": str(result),
        "predictions": str(predictions),
        "max_refine_fraction": 0.5,
    }
    assert evaluation_state(spec, ["a", "b"]) == "INVALID"


def test_audit_stage_is_checked(tmp_path):
    raw = tmp_path / "base.raw.jsonl"
    write_audited(raw, "scored")
    assert not audit_passes(raw, "raw")


def test_execution_snapshot_requires_all_macros(tmp_path):
    cal_input = tmp_path / "calibration.jsonl"
    test_input = tmp_path / "test.jsonl"
    write_input(cal_input, ["x"])
    write_input(test_input, ["x"])
    raw_cal = tmp_path / "cal.raw.jsonl"
    scored_cal = tmp_path / "cal.scored.jsonl"
    raw_test = tmp_path / "test.raw.jsonl"
    scored_test = tmp_path / "test.scored.jsonl"
    for path, stage in (
        (raw_cal, "raw"),
        (scored_cal, "scored"),
        (raw_test, "raw"),
        (scored_test, "scored"),
    ):
        write_audited(path, stage)

    result = tmp_path / "cap.json"
    predictions = tmp_path / "cap.predictions.jsonl"
    result.write_text(
        json.dumps(
            {
                "protocol": "trajuq_adaptive_evaluation_v2",
                "execution_semantics": "selective_second_pass_rerun",
                "max_refine_fraction": 0.5,
                "n_test": 1,
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text('{"example_id":"x"}\n', encoding="utf-8")

    macro = tmp_path / "macro.json"
    plan = {
        "expected_cells": 1,
        "fractions": [{"key": "cap_050", "value": 0.5, "primary": True}],
        "macro_aggregation": {
            "cap_050": {
                "output": str(macro),
                "argv": ["python", "aggregate.py"],
                "max_refine_fraction": 0.5,
                "primary": True,
            }
        },
        "cells": [
            {
                "cell": "llada:gsm8k",
                "adapter": "llada",
                "dataset": "gsm8k",
                "calibration_base": {
                    "input": str(cal_input),
                    "raw_output": str(raw_cal),
                    "scored_output": str(scored_cal),
                },
                "test_base": {
                    "input": str(test_input),
                    "raw_output": str(raw_test),
                    "scored_output": str(scored_test),
                },
                "evaluations": {
                    "cap_050": {
                        "result": str(result),
                        "predictions": str(predictions),
                        "max_refine_fraction": 0.5,
                    }
                },
            }
        ],
    }

    snapshot = execution_snapshot(plan)
    assert snapshot["complete_cells"] == 1
    assert snapshot["status"] == "IN_PROGRESS"

    macro.write_text(
        json.dumps(
            {
                "protocol": "trajuq_adaptive_macro_v1",
                "execution_semantics": "selective_second_pass_rerun",
                "cell_set_complete": True,
                "expected_cells": 1,
                "observed_cells": 1,
                "fraction_key": "cap_050",
                "max_refine_fraction": 0.5,
                "is_primary_fraction": True,
            }
        ),
        encoding="utf-8",
    )
    assert execution_snapshot(plan)["status"] == "COMPLETE"


def test_adaptive_provenance_rejects_refined_input_drift(
    tmp_path,
    monkeypatch,
):
    commit = "c" * 40
    primary_plan = tmp_path / "primary_plan.json"
    primary_config = tmp_path / "primary.yaml"
    generation_input = tmp_path / "calibration.jsonl"
    refined = tmp_path / "refined.scored.jsonl"
    refined_audit = tmp_path / "refined.scored.jsonl.audit.json"

    primary_plan.write_text('{"status":"PASS"}\n', encoding="utf-8")
    primary_config.write_text("experiment:\n  seed: 2027\n", encoding="utf-8")
    generation_input.write_text('{"id":"x"}\n', encoding="utf-8")
    refined.write_text('{"example_id":"x"}\n', encoding="utf-8")
    refined_audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": commit,
                "git_clean": True,
                "stage": "scored",
                "sha256": _sha256(refined),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    partition = {
        "generation_input": str(generation_input),
        "generation_input_sha256": _sha256(generation_input),
        "refined_scored": str(refined),
        "refined_scored_sha256": _sha256(refined),
        "refined_audit": str(refined_audit),
        "refined_audit_sha256": _sha256(refined_audit),
    }
    plan = {
        "protocol": "trajuq_adaptive_matrix_plan_v3",
        "status": "PASS",
        "experiment_git_commit": commit,
        "primary_plan": str(primary_plan),
        "primary_plan_sha256": _sha256(primary_plan),
        "primary_config": str(primary_config),
        "primary_config_sha256": _sha256(primary_config),
        "cells": [
            {
                "cell": "llada:gsm8k",
                "calibration_base": {
                    "input": str(generation_input),
                    "input_sha256": _sha256(generation_input),
                },
                "test_base": {
                    "input": str(generation_input),
                    "input_sha256": _sha256(generation_input),
                },
                "provenance": {
                    "calibration": dict(partition),
                    "test": dict(partition),
                },
            }
        ],
    }
    monkeypatch.setattr(execute_adaptive_plan, "git_head", lambda: commit)
    monkeypatch.setattr(
        execute_adaptive_plan,
        "git_status_porcelain",
        lambda: "",
    )
    monkeypatch.setattr(
        execute_adaptive_plan,
        "validate_primary_plan_provenance",
        lambda payload, *, require_current_commit=True: [],
    )

    assert validate_plan_provenance(plan) == []

    refined.write_text('{"example_id":"tampered"}\n', encoding="utf-8")
    failures = validate_plan_provenance(plan)
    assert any("refined scored input hash drifted" in item for item in failures)
    assert any("refined audit no longer binds input" in item for item in failures)


def test_adaptive_audit_rejects_other_commit(tmp_path):
    raw = tmp_path / "base.raw.jsonl"
    write_audited(raw, "raw", commit="other-commit")
    assert not audit_passes(raw, "raw", "frozen-commit")


def test_adaptive_macro_rejects_consumed_source_drift(tmp_path):
    commit = "frozen-commit"
    plan_path = tmp_path / "adaptive_plan.json"
    refined = tmp_path / "refined.scored.jsonl"
    result = tmp_path / "cap.result.json"
    predictions = tmp_path / "cap.predictions.jsonl"
    macro = tmp_path / "macro.json"

    plan_path.write_text('{"status":"PASS"}\n', encoding="utf-8")
    refined.write_text('{"example_id":"x"}\n', encoding="utf-8")
    result.write_text('{"protocol":"trajuq_adaptive_evaluation_v2"}\n', encoding="utf-8")
    predictions.write_text('{"example_id":"x"}\n', encoding="utf-8")

    plan = {
        "protocol": "trajuq_adaptive_matrix_plan_v3",
        "experiment_git_commit": commit,
        "_plan_path": str(plan_path),
        "cells": [
            {
                "cell": "llada:gsm8k",
                "provenance": {
                    "test": {"refined_scored": str(refined)},
                },
                "evaluations": {
                    "cap_050": {
                        "result": str(result),
                        "predictions": str(predictions),
                    }
                },
            }
        ],
    }
    spec = {
        "output": str(macro),
        "fraction_key": "cap_050",
        "max_refine_fraction": 0.5,
        "primary": True,
    }
    sources = {
        "plan": plan_path,
        "refined_scored:llada:gsm8k": refined,
        "adaptive_result:llada:gsm8k": result,
        "adaptive_predictions:llada:gsm8k": predictions,
    }
    macro.write_text(
        json.dumps(
            {
                "protocol": "trajuq_adaptive_macro_v1",
                "execution_semantics": "selective_second_pass_rerun",
                "cell_set_complete": True,
                "expected_cells": 1,
                "observed_cells": 1,
                "fraction_key": "cap_050",
                "max_refine_fraction": 0.5,
                "is_primary_fraction": True,
                "provenance": {
                    "protocol": "trajuq_analysis_source_binding_v1",
                    "git_commit": commit,
                    "git_clean": True,
                    "sources": {
                        name: {
                            "path": str(path),
                            "sha256": _sha256(path),
                        }
                        for name, path in sources.items()
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    assert macro_state(plan, spec, 1) == "COMPLETE"
    predictions.write_text('{"example_id":"changed"}\n', encoding="utf-8")
    assert macro_state(plan, spec, 1) == "INVALID"


def test_adaptive_evaluation_state_rejects_scored_source_drift(tmp_path):
    commit = "frozen-commit"
    source_paths = {}
    source_sha = {}
    for name in ("cal_base", "cal_refined", "test_base", "test_refined"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text('{"example_id":"x"}\n', encoding="utf-8")
        source_paths[name] = str(path)
        source_sha[name] = _sha256(path)

    provenance = {
        "protocol": "trajuq_analysis_source_binding_v1",
        "git_commit": commit,
        "git_clean": True,
        "sources": {
            name: {
                "path": path,
                "sha256": source_sha[name],
            }
            for name, path in source_paths.items()
        },
    }
    compact = {
        "protocol": "trajuq_analysis_source_binding_v1",
        "git_commit": commit,
        "git_clean": True,
        "source_sha256": source_sha,
    }

    result = tmp_path / "result.json"
    predictions = tmp_path / "predictions.jsonl"
    result.write_text(
        json.dumps(
            {
                "protocol": "trajuq_adaptive_evaluation_v2",
                "execution_semantics": "selective_second_pass_rerun",
                "max_refine_fraction": 0.5,
                "n_test": 1,
                "provenance": provenance,
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            {
                "example_id": "x",
                "analysis_provenance": compact,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = {
        "result": str(result),
        "predictions": str(predictions),
        "max_refine_fraction": 0.5,
    }

    assert (
        evaluation_state(
            spec,
            ["x"],
            source_paths,
            commit,
        )
        == "COMPLETE"
    )

    Path(source_paths["test_refined"]).write_text(
        '{"example_id":"changed"}\n',
        encoding="utf-8",
    )
    assert (
        evaluation_state(
            spec,
            ["x"],
            source_paths,
            commit,
        )
        == "INVALID"
    )


def test_adaptive_provenance_surfaces_nested_primary_failure(
    tmp_path,
    monkeypatch,
):
    commit = "f" * 40
    primary_plan = tmp_path / "primary_plan.json"
    primary_config = tmp_path / "primary.yaml"
    primary_plan.write_text('{"status":"PASS"}\n', encoding="utf-8")
    primary_config.write_text("experiment:\n  seed: 2027\n", encoding="utf-8")

    plan = {
        "protocol": "trajuq_adaptive_matrix_plan_v3",
        "status": "PASS",
        "experiment_git_commit": commit,
        "primary_plan": str(primary_plan),
        "primary_plan_sha256": _sha256(primary_plan),
        "primary_config": str(primary_config),
        "primary_config_sha256": _sha256(primary_config),
        "cells": [],
    }
    monkeypatch.setattr(execute_adaptive_plan, "git_head", lambda: commit)
    monkeypatch.setattr(
        execute_adaptive_plan,
        "git_status_porcelain",
        lambda: "",
    )
    monkeypatch.setattr(
        execute_adaptive_plan,
        "validate_primary_plan_provenance",
        lambda payload, *, require_current_commit=True: [
            "primary readiness bound source hash drifted"
        ],
    )

    failures = validate_plan_provenance(plan)
    assert any(
        "primary plan provenance: primary readiness bound source hash drifted"
        in item
        for item in failures
    )
