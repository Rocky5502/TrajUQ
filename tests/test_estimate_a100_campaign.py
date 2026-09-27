from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.estimate_a100_campaign import (
    base_budget_ratio,
    make_markdown,
    repeated_prepared_example_count,
    validate_timing_source,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, adapter: str, times: list[float]) -> None:
    rows = []
    for index, seconds in enumerate(times):
        rows.append(
            {
                "example_id": f"x:{index}",
                "adapter": adapter,
                "model": f"example/{adapter}",
                "metadata": {
                    "source_git_commit": "abc",
                    "source_git_clean": True,
                },
                "generation": {
                    "metadata": {
                        "model_revision": "model-sha",
                        "wall_time_seconds": seconds,
                        "peak_gpu_memory_bytes": 1024 + index,
                    }
                },
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_audit(
    path: Path,
    source: Path,
    *,
    commit: str,
    status: str = "PASS",
    clean: bool = True,
) -> None:
    payload = {
        "protocol": "trajuq_run_audit_v1",
        "status": status,
        "stage": "raw",
        "git_commit": commit,
        "git_clean": clean,
        "sha256": sha256_file(source),
        "source_git_commits": [commit],
        "source_git_clean_values": [True],
    }
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def test_validate_timing_source_summarizes_audited_trace(tmp_path):
    raw = tmp_path / "llada.raw.jsonl"
    audit = raw.with_suffix(raw.suffix + ".audit.json")
    write_jsonl(raw, "llada", [1.0, 2.0, 3.0, 4.0, 5.0])
    write_audit(audit, raw, commit="abc")

    result = validate_timing_source(
        "llada",
        raw,
        expected_commit="abc",
        expected_model_id="example/llada",
        expected_model_revision="model-sha",
    )

    assert result["records"] == 5
    assert result["source_git_commit"] == "abc"
    assert result["source_git_clean"] is True
    assert result["model_id"] == "example/llada"
    assert result["model_revision"] == "model-sha"
    assert result["wall_time_seconds"]["mean"] == pytest.approx(3.0)
    assert result["wall_time_seconds"]["p50"] == pytest.approx(3.0)
    assert result["wall_time_seconds"]["p90"] == pytest.approx(4.6)
    assert result["examples_per_hour"]["mean"] == pytest.approx(1200.0)
    assert result["peak_gpu_memory_bytes"] == 1028


def test_validate_timing_source_rejects_stale_source_hash(tmp_path):
    raw = tmp_path / "dream.raw.jsonl"
    audit = raw.with_suffix(raw.suffix + ".audit.json")
    write_jsonl(raw, "dream", [1.0, 1.5])
    write_audit(audit, raw, commit="abc")
    raw.write_text(
        raw.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="audit SHA"):
        validate_timing_source(
            "dream",
            raw,
            expected_commit="abc",
        )


def test_validate_timing_source_rejects_wrong_commit(tmp_path):
    raw = tmp_path / "sdar.raw.jsonl"
    audit = raw.with_suffix(raw.suffix + ".audit.json")
    write_jsonl(raw, "sdar", [0.4, 0.5])
    rows = [
        json.loads(line)
        for line in raw.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        row["metadata"]["source_git_commit"] = "old"
    raw.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_audit(audit, raw, commit="old")

    with pytest.raises(ValueError, match="commit differs"):
        validate_timing_source(
            "sdar",
            raw,
            expected_commit="new",
        )


def test_base_budget_ratio_uses_model_specific_step_fields():
    cfg = {
        "adaptive_compute": {
            "budgets": {
                "llada": {"base_steps": 32, "refined_steps": 128},
                "dream": {"base_steps": 32, "refined_steps": 128},
                "sdar": {
                    "base_denoising_steps": 2,
                    "refined_denoising_steps": 4,
                },
            }
        }
    }

    assert base_budget_ratio(cfg, "llada") == pytest.approx(0.25)
    assert base_budget_ratio(cfg, "dream") == pytest.approx(0.25)
    assert base_budget_ratio(cfg, "sdar") == pytest.approx(0.5)


def test_markdown_labels_estimate_as_planning_only():
    payload = {
        "experiment_git_commit": "a" * 40,
        "counts": {
            "primary_examples": 100,
            "rq2_auxiliary_generations": 500,
            "rq3_base_generations": 100,
            "total_generation_calls": 700,
        },
        "per_model": [
            {
                "adapter": "llada",
                "timing": {
                    "records": 50,
                    "wall_time_seconds": {"mean": 2.0},
                },
                "mean_projection_hours": {
                    "primary": 1.0,
                    "rq2_repeated": 5.0,
                    "rq3_base": 0.25,
                    "total": 6.25,
                },
            }
        ],
        "aggregate": {
            "mean_projection_hours": 6.25,
            "p90_planning_hours": 8.0,
            "mean_projection_days_24h": 6.25 / 24.0,
            "p90_planning_days_24h": 8.0 / 24.0,
        },
    }

    text = make_markdown(payload)

    assert "Planning estimate only" in text
    assert "not an empirical paper result" in text
    assert "700" in text
    assert "6.25 A100-hours" in text
    assert "0.26 days" in text
    assert "0.33 days" in text
    assert "not a statistical 90% confidence bound" in text


def test_validate_timing_source_rejects_dirty_row_provenance(tmp_path):
    raw = tmp_path / "llada.raw.jsonl"
    audit = raw.with_suffix(raw.suffix + ".audit.json")
    write_jsonl(raw, "llada", [1.0, 2.0])
    rows = [
        json.loads(line)
        for line in raw.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["metadata"]["source_git_clean"] = False
    raw.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_audit(audit, raw, commit="abc")

    with pytest.raises(ValueError, match="not generated from a clean checkout"):
        validate_timing_source(
            "llada",
            raw,
            expected_commit="abc",
        )


def test_validate_timing_source_rejects_wrong_frozen_model(tmp_path):
    raw = tmp_path / "sdar.raw.jsonl"
    audit = raw.with_suffix(raw.suffix + ".audit.json")
    write_jsonl(raw, "sdar", [0.4, 0.5])
    write_audit(audit, raw, commit="abc")

    with pytest.raises(ValueError, match="model ID"):
        validate_timing_source(
            "sdar",
            raw,
            expected_commit="abc",
            expected_model_id="JetLM/SDAR-1.7B-Chat",
            expected_model_revision="model-sha",
        )


def test_validate_timing_source_rejects_wrong_frozen_revision(tmp_path):
    raw = tmp_path / "dream.raw.jsonl"
    audit = raw.with_suffix(raw.suffix + ".audit.json")
    write_jsonl(raw, "dream", [0.8, 0.9])
    write_audit(audit, raw, commit="abc")

    with pytest.raises(ValueError, match="model revision"):
        validate_timing_source(
            "dream",
            raw,
            expected_commit="abc",
            expected_model_id="example/dream",
            expected_model_revision="other-sha",
        )


def test_repeated_prepared_count_uses_only_frozen_eligible_datasets():
    counts = {
        "gsm8k": {"total": 1000},
        "math500": {"total": 500},
        "gpqa_diamond": {"total": 198},
        "arc_challenge": {"total": 799},
        "mmlu_pro": {"total": 570},
        "truthfulqa": {"total": 817},
    }

    assert repeated_prepared_example_count(
        counts,
        ["gsm8k", "math500", "gpqa_diamond", "mmlu_pro"],
    ) == 2268


def test_repeated_prepared_count_rejects_missing_eligible_dataset():
    with pytest.raises(ValueError, match="missing from prepared counts"):
        repeated_prepared_example_count(
            {"gsm8k": {"total": 1000}},
            ["gsm8k", "math500"],
        )
