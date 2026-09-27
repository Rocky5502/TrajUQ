from scripts.compare_run_parity import (
    environment_provenance_errors,
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
        "metadata": {
            "source_git_commit": "frozen-commit",
            "source_git_clean": True,
        },
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
                "torch_version": "2.5.1+cu124",
                "transformers_version": "4.52.4",
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
        "torch_version": "2.5.1+cu124",
        "transformers_version": "4.52.4",
        "source_git_commit": "frozen-commit",
        "source_git_clean": True,
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


def test_record_provenance_rejects_stale_source_commit():
    reference = row()
    instrumented = row()
    instrumented["metadata"]["source_git_commit"] = "other-commit"

    errors = provenance_mismatches(
        "example-1",
        reference,
        instrumented,
        require=True,
    )

    assert any(
        item["field"] == "metadata.source_git_commit"
        and item["reason"] == "provenance_mismatch"
        for item in errors
    )


def test_record_provenance_rejects_dirty_source_checkout():
    reference = row()
    instrumented = row()
    instrumented["metadata"]["source_git_clean"] = False

    errors = provenance_mismatches(
        "example-1",
        reference,
        instrumented,
        require=True,
    )

    assert any(
        item["reason"] == "source_checkout_not_clean"
        for item in errors
    )


def test_environment_provenance_rejects_stale_source_commit():
    errors = environment_provenance_errors(
        {
            "source_git_commit": "old-commit",
            "source_git_clean": True,
        },
        commit="frozen-commit",
        clean=True,
        require=True,
    )
    assert any(
        item["reason"] == "source_git_commit_differs_from_comparator"
        for item in errors
    )


def test_environment_provenance_accepts_matching_clean_source():
    assert environment_provenance_errors(
        {
            "source_git_commit": "frozen-commit",
            "source_git_clean": True,
        },
        commit="frozen-commit",
        clean=True,
        require=True,
    ) == []


def test_runtime_provenance_rejects_transformers_mismatch():
    reference = row()
    instrumented = row()
    instrumented["generation"]["metadata"]["transformers_version"] = "4.53.0"

    errors = provenance_mismatches(
        "example-1",
        reference,
        instrumented,
        require=True,
    )

    assert any(
        item["field"] == "generation.metadata.transformers_version"
        and item["reason"] == "provenance_mismatch"
        for item in errors
    )
