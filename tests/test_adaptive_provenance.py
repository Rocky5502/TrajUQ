import numpy as np
import pytest

from scripts.evaluate_adaptive import (
    measured_second_pass_cost,
    validate_pair_provenance,
)


def record(*, steps: int, seed: int = 2027, revision: str = "sha") -> dict:
    return {
        "example_id": "x",
        "dataset": "gsm8k",
        "model": "model",
        "adapter": "llada",
        "seed": seed,
        "prompt_sha256": "prompt",
        "schema_version": "trajuq-record-v4",
        "correct": True,
        "generation": {
            "features": {},
            "metadata": {
                "model_revision": revision,
                "steps_requested": steps,
                "gen_length": 128,
                "block_length": 128,
                "temperature": 0.0,
                "cfg_scale": 0.0,
                "remasking": "low_confidence",
            },
        },
        "metadata": {
            "model_revision": revision,
            "scorer": "gsm8k_numeric_exact",
            "scoring_input_sha256": "inputsha",
        },
    }


def test_pair_provenance_allows_only_compute_budget_difference():
    validate_pair_provenance("x", record(steps=32), record(steps=128))


def test_pair_provenance_rejects_seed_mismatch():
    with pytest.raises(SystemExit):
        validate_pair_provenance(
            "x",
            record(steps=32, seed=2027),
            record(steps=128, seed=2028),
        )


def test_second_pass_measured_cost_is_conservative():
    risk = np.array([0.1, 0.9])
    base = np.array([2.0, 4.0])
    refined = np.array([8.0, 10.0])
    result = measured_second_pass_cost(risk, 0.5, base, refined)
    assert result is not None
    assert result["base_only_mean"] == 3.0
    assert result["adaptive_second_pass_mean"] == 8.0
    assert result["random_second_pass_expected_mean"] == 7.5
