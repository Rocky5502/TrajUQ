from scripts.audit_run import audit_run_provenance


def base_row(**overrides):
    row = {
        "dataset": "gsm8k",
        "prompt_sha256": "a" * 64,
        "model": "org/model",
        "adapter": "llada",
        "seed": 2027,
        "metadata": {
            "model_revision": "model-sha",
            "source_git_commit": "source-sha",
            "source_git_clean": True,
        },
        "generation": {
            "metadata": {"model_revision": "model-sha"},
        },
    }
    row.update(overrides)
    return row


def test_run_provenance_accepts_one_coherent_run():
    failures, provenance = audit_run_provenance(
        [base_row(), base_row()],
        require_revision=True,
    )

    assert failures == []
    assert provenance == {
        "datasets": ["gsm8k"],
        "models": ["org/model"],
        "adapters": ["llada"],
        "seeds": [2027],
        "model_revisions": ["model-sha"],
        "source_git_commits": ["source-sha"],
        "source_git_clean_values": [True],
    }


def test_run_provenance_rejects_mixed_dataset_seed_and_revision():
    other = base_row(
        dataset="math500",
        seed=2028,
        metadata={
            "model_revision": "other-sha",
            "source_git_commit": "source-sha",
            "source_git_clean": True,
        },
        generation={"metadata": {"model_revision": "other-sha"}},
    )
    failures, _ = audit_run_provenance(
        [base_row(), other],
        require_revision=True,
    )

    assert any("exactly one dataset" in item for item in failures)
    assert any("exactly one seed" in item for item in failures)
    assert any("exactly one model revision" in item for item in failures)


def test_run_provenance_rejects_record_generation_revision_disagreement():
    row = base_row(
        metadata={
            "model_revision": "record-sha",
            "source_git_commit": "source-sha",
            "source_git_clean": True,
        },
        generation={"metadata": {"model_revision": "generation-sha"}},
    )
    failures, _ = audit_run_provenance(
        [row],
        require_revision=True,
    )

    assert any("revisions disagree" in item for item in failures)
    assert any("exactly one model revision" in item for item in failures)


def test_run_provenance_rejects_missing_or_malformed_prompt_hash():
    missing = base_row()
    missing.pop("prompt_sha256")
    malformed = base_row(prompt_sha256="not-a-sha")

    failures, _ = audit_run_provenance(
        [missing, malformed],
        require_revision=True,
    )

    assert sum(
        "prompt_sha256 must be a 64-character hexadecimal digest" in item
        for item in failures
    ) == 2


def test_run_provenance_requires_revision_when_requested():
    row = base_row(
        metadata={
            "source_git_commit": "source-sha",
            "source_git_clean": True,
        },
        generation={"metadata": {}},
    )

    failures, provenance = audit_run_provenance(
        [row],
        require_revision=True,
    )

    assert any("model revision is not pinned" in item for item in failures)
    assert any("requires exactly one pinned model revision" in item for item in failures)
    assert provenance["model_revisions"] == []


def test_run_provenance_requires_adapter_reported_revision():
    row = base_row(
        metadata={
            "model_revision": "model-sha",
            "source_git_commit": "source-sha",
            "source_git_clean": True,
        },
        generation={"metadata": {}},
    )

    failures, _ = audit_run_provenance(
        [row],
        require_revision=True,
    )

    assert any(
        "generation metadata model revision is not pinned" in item
        for item in failures
    )
