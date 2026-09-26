from pathlib import Path


def test_remaining_parity_wrapper_binds_frozen_state():
    text = Path("scripts/parity_remaining_windows.ps1").read_text(encoding="utf-8")

    assert "trajuq_revision_ledger_v2" in text
    assert "trajuq_cpu_verification_v1" in text
    assert "git status --porcelain" in text
    assert "results/cpu_verification.json" in text
    assert "results/revision_ledger.json" in text
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
