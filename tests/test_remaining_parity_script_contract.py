from pathlib import Path


def test_remaining_parity_wrapper_binds_frozen_state():
    text = Path("scripts/parity_remaining_windows.ps1").read_text(encoding="utf-8")

    assert "trajuq_revision_ledger_v2" in text
    assert "trajuq_cpu_verification_v1" in text
    assert "git status --porcelain" in text
    assert "results/cpu_verification.json" in text
    assert "results/revision_ledger.json" in text
    assert "results/gated_dataset_access.json" in text
    assert "trajuq_gated_dataset_access_v1" in text
    assert "revision_ledger_sha256" in text
    assert "data/processed/gsm8k/calibration.jsonl" in text


def test_remaining_parity_wrapper_enforces_exact_v3_and_non_overwrite():
    text = Path("scripts/parity_remaining_windows.ps1").read_text(encoding="utf-8")

    assert "trajuq_exact_parity_v3" in text
    assert "--require-token-ids" in text
    assert "--require-provenance" in text
    assert "$Parity.shared_records -eq 20" in text
    assert "$Parity.mismatch_count -eq 0" in text
    assert "$Parity.provenance_mismatch_count -eq 0" in text
    assert "Require-No-PartialArtifacts" in text
    assert "will not overwrite evidence" in text


def test_remaining_parity_wrapper_preserves_native_sdar_and_dream_settings():
    text = Path("scripts/parity_remaining_windows.ps1").read_text(encoding="utf-8")

    assert "run_sdar_reference.py" in text
    assert "--temperature 1.0" in text
    assert "--denoising-steps 4" in text
    assert "--sdar-denoising-steps 4" in text
    assert "--remasking-strategy low_confidence_dynamic" in text
    assert "--sdar-remasking-strategy low_confidence_dynamic" in text

    assert "run_dream_reference.py" in text
    assert "--steps 128" in text
    assert "--temperature 0" in text
    assert "--alg entropy" in text
    assert "--dream-alg entropy" in text


def test_remaining_parity_wrapper_writes_expected_artifacts():
    text = Path("scripts/parity_remaining_windows.ps1").read_text(encoding="utf-8")

    assert "results/sdar_parity.json" in text
    assert "results/dream_parity.json" in text
    assert "results/sdar_instrumentation_overhead.json" in text
    assert "results/dream_instrumentation_overhead.json" in text


def test_remaining_parity_wrapper_revalidates_source_and_split_hashes():
    text = Path("scripts/parity_remaining_windows.ps1").read_text(encoding="utf-8")

    assert "$Parity.git_clean -eq $true" in text
    assert "reference_sha256" in text
    assert "instrumented_sha256" in text
    assert "calibration_sha256" in text
    assert "test_sha256" in text
    assert "split_manifest.json" in text
    assert "split_audit.json" in text
    assert "zero-overlap PASS" in text
    assert "trajuq_split_pair_audit_v1" in text
    assert "source_git_commit" in text
    assert "source_git_clean" in text
    assert "$Audit.requested_revision -ne $DatasetSha" in text
    assert "function Test-RunAuditPass" in text
    assert '$InstrumentedAuditOk = Test-RunAuditPass $Instrumented "raw"' in text
    assert "$InstrumentedAuditOk" in text


def test_wrappers_use_shared_frozen_preprimary_validator():
    text = Path("scripts/parity_remaining_windows.ps1").read_text(encoding="utf-8")
    assert "scripts/validate_frozen_preprimary.py" in text
    text = Path("scripts/parity_remaining_linux.sh").read_text(encoding="utf-8")
    assert "scripts/validate_frozen_preprimary.py" in text


def test_linux_remaining_parity_rejects_split_from_other_commit():
    text = Path("scripts/parity_remaining_linux.sh").read_text(encoding="utf-8")

    assert 'meta.get("source_git_clean") is True' in text
    assert 'meta.get("source_git_commit")=="$FROZEN_COMMIT"' in text
    assert 'audit.get("source_git_clean") is True' in text
    assert 'audit.get("source_git_commit")=="$FROZEN_COMMIT"' in text


def test_remaining_parity_wrappers_preflight_exact_frozen_models():
    windows = Path("scripts/parity_remaining_windows.ps1").read_text(
        encoding="utf-8"
    )
    linux = Path("scripts/parity_remaining_linux.sh").read_text(
        encoding="utf-8"
    )

    for text in (windows, linux):
        assert "scripts/preflight.py" in text
        assert "results/frozen_models/sdar.yaml" in text
        assert "results/frozen_models/dream.yaml" in text
        assert "--require-cuda" in text


def test_remaining_parity_wrappers_require_source_bound_v3_reuse():
    windows = Path("scripts/parity_remaining_windows.ps1").read_text(
        encoding="utf-8"
    )
    linux = Path("scripts/parity_remaining_linux.sh").read_text(
        encoding="utf-8"
    )

    assert "$Parity.run_provenance.source_git_commit -eq $Head" in windows
    assert "$Parity.run_provenance.source_git_clean -eq $true" in windows
    assert 'p.get("run_provenance",{}).get("source_git_commit")==commit' in linux
    assert 'p.get("run_provenance",{}).get("source_git_clean") is True' in linux
