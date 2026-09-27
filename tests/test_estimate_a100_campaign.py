from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.estimate_a100_campaign import (
    base_budget_ratio,
    make_markdown,
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
                "generation": {
                    "metadata": {
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
    )

    assert result["records"] == 5
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
        },
    }

    text = make_markdown(payload)

    assert "Planning estimate only" in text
    assert "not an empirical paper result" in text
    assert "6.25 A100-hours" in text
    assert "not a statistical 90% confidence bound" in text
