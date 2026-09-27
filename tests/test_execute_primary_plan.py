import hashlib
import json
from pathlib import Path

from trajuq.io import sha256_text
from trajuq.schema import SCHEMA_VERSION

from scripts.execute_primary_plan import (
    artifact_id_state,
    audit_passes,
    command_needed,
    execution_snapshot,
    partition_state,
    validate_plan_provenance,
)


def write_artifact(path: Path, content: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_input(path: Path, ids: list[str]) -> None:
    write_artifact(
        path,
        "".join(json.dumps({"id": item_id}) + "\n" for item_id in ids),
    )


def write_run(path: Path, ids: list[str]) -> None:
    write_artifact(
        path,
        "".join(
            json.dumps({"example_id": item_id}) + "\n"
            for item_id in ids
        ),
    )


def write_pass_audit(
    path: Path,
    *,
    stage: str | None = None,
    commit: str = "frozen-commit",
) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = path.with_suffix(path.suffix + ".audit.json")
    resolved_stage = stage or ("scored" if "scored" in path.name else "raw")
    audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": commit,
                "git_clean": True,
                "stage": resolved_stage,
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )


def test_audit_passes_requires_matching_hash(tmp_path):
    artifact = tmp_path / "run.jsonl"
    write_artifact(artifact)
    write_pass_audit(artifact)
    assert audit_passes(artifact, "raw")

    artifact.write_text("changed\n", encoding="utf-8")
    assert not audit_passes(artifact, "raw")


def test_partition_complete_only_after_both_audited_artifacts(tmp_path):
    source = tmp_path / "calibration.jsonl"
    raw = tmp_path / "cal.raw.jsonl"
    scored = tmp_path / "cal.scored.jsonl"
    write_input(source, ["a", "b"])
    spec = {
        "input": str(source),
        "raw_output": str(raw),
        "scored_output": str(scored),
    }

    assert not partition_state(spec)["complete"]
    write_run(raw, ["a", "b"])
    write_pass_audit(raw)
    assert not partition_state(spec)["complete"]
    write_run(scored, ["a", "b"])
    write_pass_audit(scored)
    assert partition_state(spec)["complete"]


def test_complete_resident_raw_hands_off_to_canonical_audit_and_score(tmp_path):
    source = tmp_path / "input.jsonl"
    raw = tmp_path / "run.raw.jsonl"
    scored = tmp_path / "run.scored.jsonl"
    write_input(source, ["a", "b"])
    write_run(raw, ["a", "b"])
    spec = {
        "input": str(source),
        "raw_output": str(raw),
        "scored_output": str(scored),
    }

    assert artifact_id_state(raw, spec, "frozen-commit") == "COMPLETE"
    assert not command_needed("generate", spec, "frozen-commit")
    assert command_needed("audit_raw", spec, "frozen-commit")
    assert command_needed("score", spec, "frozen-commit")
    assert command_needed("audit_scored", spec, "frozen-commit")
    assert not partition_state(spec, "frozen-commit")["complete"]


def test_execution_snapshot_counts_complete_partitions(tmp_path):
    cal_input = tmp_path / "calibration.jsonl"
    test_input = tmp_path / "test.jsonl"
    cal_raw = tmp_path / "cal.raw.jsonl"
    cal_scored = tmp_path / "cal.scored.jsonl"
    write_input(cal_input, ["c"])
    write_input(test_input, ["t"])
    for path in (cal_raw, cal_scored):
        write_run(path, ["c"])
        write_pass_audit(path)

    plan = {
        "protocol": "trajuq_primary_matrix_plan_v1",
        "expected_cells": 1,
        "experiment_git_commit": "frozen-commit",
        "cells": [
            {
                "cell": "llada:gsm8k",
                "adapter": "llada",
                "dataset": "gsm8k",
                "calibration": {
                    "input": str(cal_input),
                    "raw_output": str(cal_raw),
                    "scored_output": str(cal_scored),
                },
                "test": {
                    "input": str(test_input),
                    "raw_output": str(tmp_path / "test.raw.jsonl"),
                    "scored_output": str(tmp_path / "test.scored.jsonl"),
                },
            }
        ],
    }
    snapshot = execution_snapshot(plan)
    assert snapshot["complete_cells"] == 0
    assert snapshot["complete_partitions"] == 1
    assert snapshot["total_partitions"] == 2
    assert snapshot["status"] == "IN_PROGRESS"


def test_partial_raw_prefix_is_resumed_not_counted_complete(tmp_path):
    source = tmp_path / "input.jsonl"
    raw = tmp_path / "run.raw.jsonl"
    scored = tmp_path / "run.scored.jsonl"
    write_input(source, ["a", "b", "c"])
    write_run(raw, ["a", "b"])
    write_pass_audit(raw, stage="raw")
    spec = {
        "input": str(source),
        "raw_output": str(raw),
        "scored_output": str(scored),
    }

    assert artifact_id_state(raw, spec) == "PREFIX"
    assert command_needed("generate", spec, "frozen-commit")
    state = partition_state(spec, "frozen-commit")
    assert state["raw_id_state"] == "PREFIX"
    assert not state["complete"]


def frozen_llada_spec(tmp_path: Path) -> tuple[dict, Path]:
    source = tmp_path / "input.jsonl"
    rows = [
        {"id": "a", "dataset": "gsm8k", "prompt": "first prompt"},
        {"id": "b", "dataset": "gsm8k", "prompt": "second prompt"},
        {"id": "c", "dataset": "gsm8k", "prompt": "third prompt"},
    ]
    write_artifact(
        source,
        "".join(json.dumps(row) + "\n" for row in rows),
    )
    spec = {
        "input": str(source),
        "raw_output": str(tmp_path / "run.raw.jsonl"),
        "scored_output": str(tmp_path / "run.scored.jsonl"),
        "generate_argv": [
            "python",
            "scripts/run_generation.py",
            "--adapter",
            "llada",
            "--model",
            "example/llada",
            "--revision",
            "model-sha",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "run.raw.jsonl"),
            "--seed",
            "2027",
            "--device",
            "cuda",
            "--dtype",
            "bfloat16",
            "--steps",
            "128",
            "--max-new-tokens",
            "128",
            "--block-length",
            "128",
            "--temperature",
            "0.0",
            "--llada-cfg-scale",
            "0.0",
            "--llada-remasking",
            "low_confidence",
        ],
    }
    return spec, source


def frozen_llada_row(example_id: str, prompt: str, source: Path) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "example_id": example_id,
        "dataset": "gsm8k",
        "prompt_sha256": sha256_text(prompt),
        "model": "example/llada",
        "adapter": "llada",
        "seed": 2027,
        "generation": {
            "metadata": {
                "model_id": "example/llada",
                "model_revision": "model-sha",
                "requested_device": "cuda",
                "requested_dtype": "bfloat16",
                "steps_requested": 128,
                "gen_length": 128,
                "block_length": 128,
                "temperature": 0.0,
                "cfg_scale": 0.0,
                "remasking": "low_confidence",
            }
        },
        "metadata": {
            "source_input": str(source),
            "model_revision": "model-sha",
            "source_git_commit": "frozen-commit",
            "source_git_clean": True,
        },
    }


def test_partial_raw_resume_requires_exact_frozen_generation_provenance(tmp_path):
    spec, source = frozen_llada_spec(tmp_path)
    raw = Path(spec["raw_output"])
    row = frozen_llada_row("a", "first prompt", source)
    write_artifact(raw, json.dumps(row) + "\n")

    assert artifact_id_state(raw, spec) == "PREFIX"
    assert command_needed("generate", spec, "frozen-commit")


def test_partial_raw_resume_rejects_wrong_revision_or_sampler_settings(tmp_path):
    spec, source = frozen_llada_spec(tmp_path)
    raw = Path(spec["raw_output"])
    row = frozen_llada_row("a", "first prompt", source)

    row["generation"]["metadata"]["model_revision"] = "wrong-sha"
    write_artifact(raw, json.dumps(row) + "\n")
    assert artifact_id_state(raw, spec) == "INVALID"

    row = frozen_llada_row("a", "first prompt", source)
    row["generation"]["metadata"]["temperature"] = 0.2
    write_artifact(raw, json.dumps(row) + "\n")
    assert artifact_id_state(raw, spec) == "INVALID"


def test_partial_raw_resume_rejects_other_source_commit(tmp_path):
    spec, source = frozen_llada_spec(tmp_path)
    raw = Path(spec["raw_output"])
    row = frozen_llada_row("a", "first prompt", source)
    row["metadata"]["source_git_commit"] = "old-commit"
    write_artifact(raw, json.dumps(row) + "\n")

    assert artifact_id_state(raw, spec, "frozen-commit") == "INVALID"


def test_partial_raw_resume_rejects_wrong_prompt_binding(tmp_path):
    spec, source = frozen_llada_spec(tmp_path)
    raw = Path(spec["raw_output"])
    row = frozen_llada_row("a", "different prompt", source)
    write_artifact(raw, json.dumps(row) + "\n")

    assert artifact_id_state(raw, spec) == "INVALID"


def test_partial_scored_artifact_is_rejected_before_overwrite(tmp_path):
    source = tmp_path / "input.jsonl"
    raw = tmp_path / "run.raw.jsonl"
    scored = tmp_path / "run.scored.jsonl"
    write_input(source, ["a", "b"])
    write_run(raw, ["a", "b"])
    write_run(scored, ["a"])
    spec = {
        "input": str(source),
        "raw_output": str(raw),
        "scored_output": str(scored),
    }

    assert artifact_id_state(scored, spec) == "PREFIX"
    try:
        command_needed("score", spec, "frozen-commit")
    except ValueError as exc:
        assert "partial/invalid scored artifact" in str(exc)
    else:
        raise AssertionError("partial scored artifact must not be silently reused")


def test_validate_plan_provenance_accepts_bound_inputs(tmp_path, monkeypatch):
    primary = tmp_path / "primary.yaml"
    readiness = tmp_path / "readiness.json"
    readiness_source = tmp_path / "llada_parity.json"
    ledger = tmp_path / "revision_ledger.json"
    model = tmp_path / "llada.yaml"
    split_manifest = tmp_path / "split_manifest.json"
    split_audit = tmp_path / "split_audit.json"
    cal = tmp_path / "cal.jsonl"
    test = tmp_path / "test.jsonl"
    for path, text in (
        (primary, "experiment: {}\n"),
        (readiness_source, '{"status":"PASS"}\n'),
        (ledger, '{"status":"PASS"}\n'),
        (model, "revision: abc\n"),
        (split_manifest, '{"dataset":"gsm8k"}\n'),
        (split_audit, '{"status":"PASS"}\n'),
        (cal, '{"id":"c"}\n'),
        (test, '{"id":"t"}\n'),
    ):
        path.write_text(text, encoding="utf-8")

    readiness.write_text(
        json.dumps(
            {
                "status": "PASS",
                "source_bindings": [
                    {
                        "path": str(readiness_source),
                        "sha256": hashlib.sha256(
                            readiness_source.read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.execute_primary_plan.git_head",
        lambda: "frozen-commit",
    )
    monkeypatch.setattr(
        "scripts.execute_primary_plan.git_status_porcelain",
        lambda: "",
    )
    plan = {
        "protocol": "trajuq_primary_matrix_plan_v2",
        "experiment_git_commit": "frozen-commit",
        "primary_config": str(primary),
        "primary_config_sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
        "readiness": str(readiness),
        "readiness_sha256": hashlib.sha256(readiness.read_bytes()).hexdigest(),
        "revision_ledger": str(ledger),
        "revision_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "model_configs": {
            "llada": {
                "path": str(model),
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        },
        "cells": [
            {
                "cell": "llada:gsm8k",
                "split_manifest": str(split_manifest),
                "split_manifest_sha256": hashlib.sha256(
                    split_manifest.read_bytes()
                ).hexdigest(),
                "split_audit": str(split_audit),
                "split_audit_sha256": hashlib.sha256(
                    split_audit.read_bytes()
                ).hexdigest(),
                "calibration": {
                    "input": str(cal),
                    "input_sha256": hashlib.sha256(cal.read_bytes()).hexdigest(),
                },
                "test": {
                    "input": str(test),
                    "input_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
                },
            }
        ],
    }

    assert validate_plan_provenance(plan) == []

    readiness_source.write_text('{"status":"FAIL"}\n', encoding="utf-8")
    failures = validate_plan_provenance(plan)
    assert any("readiness bound source hash drifted" in item for item in failures)
    readiness_source.write_text('{"status":"PASS"}\n', encoding="utf-8")

    monkeypatch.setattr(
        "scripts.execute_primary_plan.git_status_porcelain",
        lambda: " M scripts/run_generation.py",
    )
    failures = validate_plan_provenance(plan)
    assert any("worktree is dirty" in item for item in failures)


def test_validate_plan_provenance_rejects_mutated_input(tmp_path, monkeypatch):
    primary = tmp_path / "primary.yaml"
    readiness = tmp_path / "readiness.json"
    ledger = tmp_path / "revision_ledger.json"
    model = tmp_path / "llada.yaml"
    split_manifest = tmp_path / "split_manifest.json"
    split_audit = tmp_path / "split_audit.json"
    cal = tmp_path / "cal.jsonl"
    test = tmp_path / "test.jsonl"
    for path in (
        primary,
        readiness,
        ledger,
        model,
        split_manifest,
        split_audit,
        cal,
        test,
    ):
        path.write_text("frozen\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.execute_primary_plan.git_head",
        lambda: "frozen-commit",
    )
    cal_hash = hashlib.sha256(cal.read_bytes()).hexdigest()
    plan = {
        "protocol": "trajuq_primary_matrix_plan_v2",
        "experiment_git_commit": "frozen-commit",
        "primary_config": str(primary),
        "primary_config_sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
        "readiness": str(readiness),
        "readiness_sha256": hashlib.sha256(readiness.read_bytes()).hexdigest(),
        "revision_ledger": str(ledger),
        "revision_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "model_configs": {
            "llada": {
                "path": str(model),
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        },
        "cells": [
            {
                "cell": "llada:gsm8k",
                "split_manifest": str(split_manifest),
                "split_manifest_sha256": hashlib.sha256(
                    split_manifest.read_bytes()
                ).hexdigest(),
                "split_audit": str(split_audit),
                "split_audit_sha256": hashlib.sha256(
                    split_audit.read_bytes()
                ).hexdigest(),
                "calibration": {
                    "input": str(cal),
                    "input_sha256": cal_hash,
                },
                "test": {
                    "input": str(test),
                    "input_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
                },
            }
        ],
    }
    cal.write_text("mutated\n", encoding="utf-8")

    failures = validate_plan_provenance(plan)
    assert any("input hash differs from plan" in item for item in failures)


def test_validate_plan_provenance_rejects_split_binding_drift(
    tmp_path,
    monkeypatch,
):
    primary = tmp_path / "primary.yaml"
    readiness = tmp_path / "readiness.json"
    ledger = tmp_path / "revision_ledger.json"
    model = tmp_path / "llada.yaml"
    split_manifest = tmp_path / "split_manifest.json"
    split_audit = tmp_path / "split_audit.json"
    cal = tmp_path / "cal.jsonl"
    test = tmp_path / "test.jsonl"
    for path in (
        primary,
        readiness,
        ledger,
        model,
        split_manifest,
        split_audit,
        cal,
        test,
    ):
        path.write_text("frozen\n", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.execute_primary_plan.git_head",
        lambda: "frozen-commit",
    )
    monkeypatch.setattr(
        "scripts.execute_primary_plan.git_status_porcelain",
        lambda: "",
    )
    plan = {
        "protocol": "trajuq_primary_matrix_plan_v2",
        "experiment_git_commit": "frozen-commit",
        "primary_config": str(primary),
        "primary_config_sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
        "readiness": str(readiness),
        "readiness_sha256": hashlib.sha256(readiness.read_bytes()).hexdigest(),
        "revision_ledger": str(ledger),
        "revision_ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "model_configs": {
            "llada": {
                "path": str(model),
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        },
        "cells": [
            {
                "cell": "llada:gsm8k",
                "split_manifest": str(split_manifest),
                "split_manifest_sha256": hashlib.sha256(
                    split_manifest.read_bytes()
                ).hexdigest(),
                "split_audit": str(split_audit),
                "split_audit_sha256": hashlib.sha256(
                    split_audit.read_bytes()
                ).hexdigest(),
                "calibration": {
                    "input": str(cal),
                    "input_sha256": hashlib.sha256(cal.read_bytes()).hexdigest(),
                },
                "test": {
                    "input": str(test),
                    "input_sha256": hashlib.sha256(test.read_bytes()).hexdigest(),
                },
            }
        ],
    }

    split_audit.write_text("mutated\n", encoding="utf-8")
    failures = validate_plan_provenance(plan)
    assert any("split_audit hash differs from plan" in item for item in failures)


def test_audit_passes_rejects_other_commit(tmp_path):
    artifact = tmp_path / "run.raw.jsonl"
    write_artifact(artifact)
    write_pass_audit(artifact, stage="raw", commit="other-commit")
    assert not audit_passes(artifact, "raw", "frozen-commit")
