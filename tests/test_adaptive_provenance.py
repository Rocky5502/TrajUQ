import numpy as np
import pytest

from scripts.evaluate_adaptive import (
    measured_second_pass_cost,
    validate_pair_provenance,
)


def record(
    *,
    steps: int,
    seed: int = 2027,
    revision: str = "sha",
    trajectory_semantics: str = "active_unresolved_positions_v4",
) -> dict:
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
                "mask_id": 126336,
                "steps_requested": steps,
                "gen_length": 128,
                "block_length": 128,
                "temperature": 0.0,
                "cfg_scale": 0.0,
                "remasking": "low_confidence",
                "trajectory_semantics": trajectory_semantics,
                "upstream_repository": "ML-GSAI/LLaDA",
                "upstream_commit": "upstream-sha",
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


def test_pair_provenance_rejects_trajectory_semantics_drift():
    with pytest.raises(SystemExit):
        validate_pair_provenance(
            "x",
            record(steps=32),
            record(
                steps=128,
                trajectory_semantics="different_semantics",
            ),
        )


def test_pair_provenance_rejects_sdar_guard_drift():
    base = record(steps=32)
    refined = record(steps=128)
    for row in (base, refined):
        row["adapter"] = "sdar"
        row["generation"]["metadata"].pop("steps_requested")
        row["generation"]["metadata"].pop("cfg_scale")
        row["generation"]["metadata"].pop("remasking")
        row["generation"]["metadata"]["block_length"] = 4
        row["generation"]["metadata"]["denoising_steps"] = (
            2 if row is base else 4
        )
        row["generation"]["metadata"]["temperature"] = 1.0
        row["generation"]["metadata"]["top_k"] = 0
        row["generation"]["metadata"]["top_p"] = 1.0
        row["generation"]["metadata"]["remasking_strategy"] = (
            "low_confidence_dynamic"
        )
        row["generation"]["metadata"]["confidence_threshold"] = 0.85
        row["generation"]["metadata"]["upstream_repository"] = "JetAstra/SDAR"
        row["generation"]["metadata"]["upstream_commit"] = "sdar-sha"
        row["generation"]["metadata"]["partial_block_transfer_guard"] = (
            "cap_requested_transfers_to_masked_positions"
        )

    validate_pair_provenance("x", base, refined)

    refined["generation"]["metadata"]["partial_block_transfer_guard"] = "disabled"
    with pytest.raises(SystemExit):
        validate_pair_provenance("x", base, refined)
