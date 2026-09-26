import hashlib
import json
from pathlib import Path

from scripts.execute_repeated_sampling import (
    audit_passes,
    execution_snapshot,
    generation_state,
    pair_state,
)


def write_raw(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"example_id":"x"}\n', encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".audit.json").write_text(
        json.dumps(
            {"status": "PASS", "stage": "raw", "sha256": digest}
        ),
        encoding="utf-8",
    )


def test_generation_state_requires_every_seed_and_manifest(tmp_path):
    argv = [
        "python",
        "scripts/run_repeated_samples.py",
        "--output-dir",
        str(tmp_path),
    ]
    write_raw(tmp_path / "seed-2027.raw.jsonl")
    write_raw(tmp_path / "seed-2028.raw.jsonl")
    assert not generation_state(argv, [2027, 2028])["complete"]

    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    assert generation_state(argv, [2027, 2028])["complete"]


def test_pair_state_rejects_partial_outputs(tmp_path):
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    metrics.write_text("{}", encoding="utf-8")
    assert pair_state(metrics, predictions) == "PARTIAL"


def test_audit_hash_must_match(tmp_path):
    path = tmp_path / "seed.raw.jsonl"
    write_raw(path)
    assert audit_passes(path)
    path.write_text("changed\n", encoding="utf-8")
    assert not audit_passes(path)


def test_execution_snapshot_waits_for_macro(tmp_path):
    cal = tmp_path / "cal"
    test = tmp_path / "test"
    for root in (cal, test):
        write_raw(root / "seed-2027.raw.jsonl")
        (root / "manifest.json").write_text("{}", encoding="utf-8")

    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    comparison = tmp_path / "comparison.json"
    metrics.write_text("{}", encoding="utf-8")
    predictions.write_text('{"example_id":"x"}\n', encoding="utf-8")
    comparison.write_text("{}", encoding="utf-8")

    macro = tmp_path / "macro.json"
    plan = {
        "expected_cells": 1,
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
                    "--output-dir",
                    str(cal),
                ],
                "test_generation_argv": [
                    "python",
                    "x",
                    "--output-dir",
                    str(test),
                ],
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
            {"cell_set_complete": True, "observed_cells": 1}
        ),
        encoding="utf-8",
    )
    assert execution_snapshot(plan)["status"] == "COMPLETE"
