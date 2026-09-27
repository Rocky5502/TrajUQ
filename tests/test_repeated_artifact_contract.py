import json
from pathlib import Path

import pytest

from scripts.aggregate_repeated_baseline import load_auxiliary_compute
from scripts.evaluate_repeated_sampling import feature_matrix, load_sample_run


def row(seed: int, temperature: float = 0.2) -> dict:
    return {
        "example_id": "x",
        "dataset": "gsm8k",
        "prompt_sha256": "abc",
        "model": "model",
        "adapter": "llada",
        "seed": seed,
        "correct": None,
        "reference": None,
        "generation": {
            "text": "FINAL: 1",
            "metadata": {
                "model_revision": "sha",
                "steps_requested": 128,
                "gen_length": 128,
                "block_length": 128,
                "temperature": temperature,
                "top_p": 1.0,
                "cfg_scale": 0.0,
                "remasking": "low_confidence",
                "mask_id": 126336,
                "trajectory_semantics": "active_unresolved_positions_v4",
                "upstream_repository": "ML-GSAI/LLaDA",
                "upstream_commit": "upstream-sha",
                "wall_time_seconds": 1.0,
            },
        },
    }


def write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in rows),
        encoding="utf-8",
    )


def test_repeated_run_signature_reads_seed_and_sampling_config(tmp_path):
    path = tmp_path / "sample.jsonl"
    write(path, [row(2027)])
    _, signature = load_sample_run(str(path))
    assert signature["seed"] == 2027
    assert signature["temperature"] == 0.2
    assert signature["model_revision"] == "sha"


def test_repeated_seed_pool_rejects_runtime_semantics_drift(tmp_path):
    first = tmp_path / "seed-2027.raw.jsonl"
    second = tmp_path / "seed-2028.raw.jsonl"
    target_row = {
        "example_id": "x",
        "dataset": "gsm8k",
        "prompt_sha256": "abc",
        "model": "model",
        "correct": True,
    }
    write(first, [row(2027)])
    changed = row(2028)
    changed["generation"]["metadata"]["upstream_commit"] = "different-upstream"
    write(second, [changed])

    with pytest.raises(SystemExit, match="generation settings differ"):
        feature_matrix(
            ["x"],
            {"x": target_row},
            [str(first), str(second)],
        )


def test_repeated_run_rejects_outcome_leakage(tmp_path):
    path = tmp_path / "sample.jsonl"
    leaked = row(2027)
    leaked["correct"] = True
    write(path, [leaked])
    with pytest.raises(SystemExit):
        load_sample_run(str(path))


def test_repeated_macro_compute_source_is_complete_and_five_sample(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "protocol": "trajuq_repeated_sampling_baseline_v2",
                "n_test": 2,
                "sample_count": 5,
                "auxiliary_compute": {
                    "test": {
                        "mean_auxiliary_wall_time_seconds_per_example": 10.0,
                        "p90_auxiliary_wall_time_seconds_per_example": 12.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "cell": "llada:gsm8k",
        "seeds": [2027, 2028, 2029, 2030, 2031],
        "repeated_metrics": str(metrics),
    }
    result = load_auxiliary_compute(spec, 2)
    assert result["sample_count"] == 5
    assert result["mean_auxiliary_wall_time_seconds_per_example"] == 10.0
    assert result["p90_auxiliary_wall_time_seconds_per_example"] == 12.0


def test_repeated_macro_compute_source_rejects_missing_wall_time(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "protocol": "trajuq_repeated_sampling_baseline_v2",
                "n_test": 1,
                "sample_count": 5,
                "auxiliary_compute": {
                    "test": {
                        "mean_auxiliary_wall_time_seconds_per_example": None,
                        "p90_auxiliary_wall_time_seconds_per_example": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    spec = {
        "cell": "llada:gsm8k",
        "seeds": [2027, 2028, 2029, 2030, 2031],
        "repeated_metrics": str(metrics),
    }
    with pytest.raises(SystemExit, match="missing/non-numeric"):
        load_auxiliary_compute(spec, 1)
