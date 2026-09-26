from scripts.compare_run_parity import (
    provenance_mismatches,
    summarize_run_provenance,
)


def row(
    *,
    adapter: str = "llada",
    seed: int = 2027,
    revision: str = "model-sha",
) -> dict:
    return {
        "dataset": "gsm8k",
        "prompt_sha256": "prompt-sha",
        "model": "example/model",
        "adapter": adapter,
        "seed": seed,
        "generation": {
            "metadata": {
                "model_revision": revision,
                "mask_id": 126336,
                "steps_requested": 128,
                "gen_length": 128,
                "block_length": 128,
                "temperature": 0.0,
                "cfg_scale": 0.0,
                "remasking": "low_confidence",
            }
        },
    }


def test_uniform_run_provenance_has_no_errors():
    reference = {"a": row(), "b": row()}
    instrumented = {"a": row(), "b": row()}

    summary, errors = summarize_run_provenance(
        ["a", "b"],
        reference,
        instrumented,
        require=True,
    )

    assert not errors
    assert summary == {
        "dataset": "gsm8k",
        "model": "example/model",
        "adapter": "llada",
        "seed": 2027,
        "model_revision": "model-sha",
    }


def test_run_provenance_rejects_mixed_seed():
    reference = {"a": row(), "b": row(seed=2028)}
    instrumented = {"a": row(), "b": row(seed=2028)}

    _, errors = summarize_run_provenance(
        ["a", "b"],
        reference,
        instrumented,
        require=True,
    )

    assert any(
        item["reason"] == "non_uniform_run_provenance"
        and item["field"] == "seed"
        for item in errors
    )


def test_run_provenance_rejects_mixed_model_revision():
    reference = {"a": row(), "b": row(revision="other-sha")}
    instrumented = {"a": row(), "b": row(revision="other-sha")}

    _, errors = summarize_run_provenance(
        ["a", "b"],
        reference,
        instrumented,
        require=True,
    )

    assert any(
        item["reason"] == "non_uniform_run_provenance"
        and item["field"] == "generation.metadata.model_revision"
        for item in errors
    )


def test_record_provenance_requires_adapter_specific_generation_fields():
    reference = row()
    instrumented = row()
    del instrumented["generation"]["metadata"]["block_length"]

    errors = provenance_mismatches(
        "example-1",
        reference,
        instrumented,
        require=True,
    )

    assert any(
        item["reason"] == "missing_provenance"
        and item["field"] == "generation.metadata.block_length"
        for item in errors
    )
