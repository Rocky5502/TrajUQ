from scripts.compare_run_parity import provenance_mismatches


def row(*, revision: str = "sha", prompt: str = "p", seed: int = 2027):
    return {
        "dataset": "gsm8k",
        "prompt_sha256": prompt,
        "model": "model",
        "adapter": "llada",
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
            }
        },
    }


def test_parity_provenance_matches_identical_runs():
    assert provenance_mismatches("x", row(), row(), require=True) == []


def test_parity_provenance_rejects_revision_mismatch():
    errors = provenance_mismatches(
        "x",
        row(revision="a"),
        row(revision="b"),
        require=True,
    )
    assert any(
        item["field"] == "generation.metadata.model_revision"
        for item in errors
    )


def test_parity_provenance_requires_prompt_hash_when_strict():
    reference = row()
    reference.pop("prompt_sha256")
    errors = provenance_mismatches(
        "x",
        reference,
        row(),
        require=True,
    )
    assert any(item["field"] == "prompt_sha256" for item in errors)



def test_strict_parity_requires_adapter_generation_settings():
    reference = row()
    reference["generation"]["metadata"].pop("block_length")
    errors = provenance_mismatches(
        "x",
        reference,
        row(),
        require=True,
    )
    assert any(
        item["field"] == "generation.metadata.block_length"
        and item["reason"] == "missing_provenance"
        for item in errors
    )


def test_strict_parity_requires_source_git_commit():
    reference = row()
    reference["metadata"].pop("source_git_commit")
    errors = provenance_mismatches(
        "x",
        reference,
        row(),
        require=True,
    )
    assert any(
        item["field"] == "metadata.source_git_commit"
        and item["reason"] == "missing_provenance"
        for item in errors
    )
