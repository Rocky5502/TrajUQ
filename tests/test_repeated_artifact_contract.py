import json
from pathlib import Path

import pytest

from scripts.evaluate_repeated_sampling import load_sample_run


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


def test_repeated_run_rejects_outcome_leakage(tmp_path):
    path = tmp_path / "sample.jsonl"
    leaked = row(2027)
    leaked["correct"] = True
    write(path, [leaked])
    with pytest.raises(SystemExit):
        load_sample_run(str(path))
