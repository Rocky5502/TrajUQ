import hashlib
import json
from pathlib import Path

import scripts.execute_repeated_sampling as execute_repeated_sampling
from scripts.execute_repeated_sampling import (
    audit_passes,
    execution_snapshot,
    generation_state,
    pair_state,
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


def write_raw(
    path: Path,
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
                "stage": "raw",
                "sha256": digest,
            }
        ),
        encoding="utf-8",
    )


def write_manifest(
    root: Path,
    input_path: Path,
    seeds: list[int],
    *,
    commit: str = "frozen-commit",
) -> None:
    payload = {
        "protocol": "repeated_sampling_v1",
        "git_commit": commit,
        "git_clean": True,
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "seeds": seeds,
        "runs": [
            {
                "seed": seed,
                "path": str(root / f"seed-{seed}.raw.jsonl"),
                "status": "GENERATED",
            }
            for seed in seeds
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_generation_state_requires_every_seed_and_manifest(tmp_path):
    source = tmp_path / "input.jsonl"
    write_input(source, ["x"])
    argv = [
        "python",
        "scripts/run_repeated_samples.py",
        "--input",
        str(source),
        "--output-dir",
        str(tmp_path),
    ]
    write_raw(tmp_path / "seed-2027.raw.jsonl")
    write_raw(tmp_path / "seed-2028.raw.jsonl")
    assert not generation_state(
        argv,
        [2027, 2028],
        "frozen-commit",
    )["complete"]

    write_manifest(tmp_path, source, [2027, 2028])
    state = generation_state(argv, [2027, 2028], "frozen-commit")
    assert state["manifest_valid"]
    assert state["expected_records"] == 1
    assert state["complete"]


def test_generation_state_rejects_audited_truncated_seed(tmp_path):
    source = tmp_path / "input.jsonl"
    write_input(source, ["a", "b"])
    argv = [
        "python",
        "scripts/run_repeated_samples.py",
        "--input",
        str(source),
        "--output-dir",
        str(tmp_path),
    ]
    write_raw(
        tmp_path / "seed-2027.raw.jsonl",
        ids=["a"],
    )
    write_manifest(tmp_path, source, [2027])

    state = generation_state(argv, [2027], "frozen-commit")
    assert not state["seeds"]["2027"]["ids_complete"]
    assert state["seeds"]["2027"]["audit_pass"]
    assert not state["complete"]


def test_pair_state_rejects_partial_outputs(tmp_path):
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    metrics.write_text("{}", encoding="utf-8")
    assert pair_state(metrics, predictions) == "PARTIAL"


def test_pair_state_rejects_incomplete_prediction_coverage(tmp_path):
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    metrics.write_text("{}", encoding="utf-8")
    predictions.write_text(
        '{"example_id":"a"}\n',
        encoding="utf-8",
    )

    assert pair_state(
        metrics,
        predictions,
        ["a", "b"],
    ) == "INVALID"


def test_audit_hash_must_match(tmp_path):
    path = tmp_path / "seed.raw.jsonl"
    write_raw(path)
    assert audit_passes(path)
    path.write_text("changed\n", encoding="utf-8")
    assert not audit_passes(path)


def test_execution_snapshot_waits_for_macro(tmp_path):
    cal = tmp_path / "cal"
    test = tmp_path / "test"
    cal_input = tmp_path / "calibration.jsonl"
    test_input = tmp_path / "test.jsonl"
    scored_target = tmp_path / "test.scored.jsonl"
    write_input(cal_input, ["x"])
    write_input(test_input, ["x"])
    scored_target.write_text('{"example_id":"x"}\n', encoding="utf-8")
    for root, source in ((cal, cal_input), (test, test_input)):
        write_raw(root / "seed-2027.raw.jsonl")
        write_manifest(root, source, [2027])

    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    comparison = tmp_path / "comparison.json"
    metrics.write_text(
        json.dumps(
            {
                "protocol": "trajuq_repeated_sampling_baseline_v2",
                "n_test": 1,
                "sample_count": 5,
                "auxiliary_compute": {
                    "test": {
                        "mean_auxiliary_wall_time_seconds_per_example": 5.0,
                        "p90_auxiliary_wall_time_seconds_per_example": 6.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text('{"example_id":"x"}\n', encoding="utf-8")
    comparison.write_text(
        json.dumps(
            {
                "protocol": "trajuq_vs_repeated_sampling_v1",
                "n": 1,
                "metrics": {"auroc": {}, "auprc": {}, "aurc": {}},
            }
        ),
        encoding="utf-8",
    )

    macro = tmp_path / "macro.json"
    plan = {
        "expected_cells": 1,
        "experiment_git_commit": "frozen-commit",
        "macro_output": str(macro),
        "cells": [
            {
                "cell": "llada:gsm8k",
                "adapter": "llada",
                "dataset": "gsm8k",
                "seeds": [2027],
                "calibration_generation_argv": [
                    "python",
                    "x",
                    "--input",
                    str(cal_input),
                    "--output-dir",
                    str(cal),
                ],
                "test_generation_argv": [
                    "python",
                    "x",
                    "--input",
                    str(test_input),
                    "--output-dir",
                    str(test),
                ],
                "provenance": {
                    "test": {
                        "scored_target": str(scored_target),
                    }
                },
                "repeated_metrics": str(metrics),
                "repeated_predictions": str(predictions),
                "direct_comparison": str(comparison),
            }
        ],
    }
    snapshot = execution_snapshot(plan)
    assert snapshot["complete_cells"] == 1
    assert snapshot["status"] == "IN_PROGRESS"

    macro.write_text(
        json.dumps(
            {
                "protocol": "trajuq_vs_repeated_macro_v2",
                "comparison": "single_trajectory_trajuq_vs_five_sample_baseline",
                "cell_set_complete": True,
                "expected_cells": 1,
                "observed_cells": 1,
                "auxiliary_compute": {
                    "aggregation": "equal_cell_weight_macro",
                    "expected_cells": 1,
                    "observed_cells_with_complete_test_wall_time": 1,
                    "complete": True,
                    "sample_count": 5,
                    "mean_test_auxiliary_wall_time_seconds_per_example": 5.0,
                    "mean_test_p90_auxiliary_wall_time_seconds_per_example": 6.0,
                },
                "macro": {
                    key: {}
                    for key in (
                        "auroc",
                        "auprc",
                        "brier",
                        "nll",
                        "ece15",
                        "aurc",
                        "risk50",
                        "risk80",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    assert execution_snapshot(plan)["status"] == "COMPLETE"


def test_repeated_provenance_rejects_primary_prediction_drift(
    tmp_path,
    monkeypatch,
):
    commit = "b" * 40
    primary_plan = tmp_path / "primary_plan.json"
    analysis_plan = tmp_path / "analysis_plan.json"
    primary_config = tmp_path / "primary.yaml"
    generation_input = tmp_path / "calibration.jsonl"
    scored = tmp_path / "calibration.scored.jsonl"
    scored_audit = tmp_path / "calibration.scored.jsonl.audit.json"
    predictions = tmp_path / "trajuq_predictions.jsonl"

    primary_plan.write_text('{"status":"PASS"}\n', encoding="utf-8")
    analysis_plan.write_text('{"status":"PASS"}\n', encoding="utf-8")
    primary_config.write_text("experiment:\n  seed: 2027\n", encoding="utf-8")
    generation_input.write_text('{"example_id":"x"}\n', encoding="utf-8")
    scored.write_text('{"example_id":"x"}\n', encoding="utf-8")
    scored_audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": commit,
                "git_clean": True,
                "stage": "scored",
                "sha256": _sha256(scored),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    predictions.write_text(
        '{"example_id":"x","risk":0.5}\n',
        encoding="utf-8",
    )

    partition = {
        "generation_input": str(generation_input),
        "generation_input_sha256": _sha256(generation_input),
        "scored_target": str(scored),
        "scored_target_sha256": _sha256(scored),
        "scored_audit": str(scored_audit),
        "scored_audit_sha256": _sha256(scored_audit),
    }
    plan = {
        "protocol": "trajuq_repeated_sampling_plan_v2",
        "status": "PASS",
        "experiment_git_commit": commit,
        "primary_plan": str(primary_plan),
        "primary_plan_sha256": _sha256(primary_plan),
        "primary_analysis_plan": str(analysis_plan),
        "primary_analysis_plan_sha256": _sha256(analysis_plan),
        "primary_config": str(primary_config),
        "primary_config_sha256": _sha256(primary_config),
        "cells": [
            {
                "cell": "llada:gsm8k",
                "provenance": {
                    "calibration": dict(partition),
                    "test": dict(partition),
                    "trajuq_predictions": str(predictions),
                    "trajuq_predictions_sha256": _sha256(predictions),
                },
            }
        ],
    }
    monkeypatch.setattr(execute_repeated_sampling, "git_head", lambda: commit)
    monkeypatch.setattr(
        execute_repeated_sampling,
        "git_status_porcelain",
        lambda: "",
    )

    assert validate_plan_provenance(plan) == []

    predictions.write_text(
        '{"example_id":"x","risk":0.9}\n',
        encoding="utf-8",
    )
    failures = validate_plan_provenance(plan)
    assert any("TrajUQ predictions hash drifted" in item for item in failures)


def test_repeated_raw_audit_rejects_other_commit(tmp_path):
    path = tmp_path / "seed.raw.jsonl"
    write_raw(path, commit="other-commit")
    assert not audit_passes(path, "frozen-commit")


def test_pair_state_rejects_missing_compute_accounting(tmp_path):
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "protocol": "trajuq_repeated_sampling_baseline_v2",
                "n_test": 1,
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text('{"example_id":"x"}\n', encoding="utf-8")
    assert pair_state(metrics, predictions, ["x"]) == "INVALID"


def test_pair_state_rejects_non_five_sample_metrics(tmp_path):
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "protocol": "trajuq_repeated_sampling_baseline_v2",
                "n_test": 1,
                "sample_count": 4,
                "auxiliary_compute": {
                    "test": {
                        "mean_auxiliary_wall_time_seconds_per_example": 4.0,
                        "p90_auxiliary_wall_time_seconds_per_example": 5.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    predictions.write_text('{"example_id":"x"}\n', encoding="utf-8")
    assert pair_state(metrics, predictions, ["x"]) == "INVALID"
