from pathlib import Path


def test_math500_wrapper_binds_frozen_state_and_llada_parity():
    text = Path("scripts/math500_gate_windows.ps1").read_text(encoding="utf-8")

    assert "trajuq_revision_ledger_v2" in text
    assert "trajuq_cpu_verification_v1" in text
    assert "git status --porcelain" in text
    assert "trajuq_exact_parity_v3" in text
    assert "$Parity.shared_records -eq 20" in text
    assert "$Parity.mismatch_count -eq 0" in text
    assert "$Parity.provenance_mismatch_count -eq 0" in text


def test_math500_wrapper_requires_frozen_scorer_dependencies():
    text = Path("scripts/math500_gate_windows.ps1").read_text(encoding="utf-8")

    assert "trajuq_math500_scorer_audit_v2" in text
    assert "$Audit.dependency_versions.'math-verify' -eq \"0.9.0\"" in text
    assert "$Audit.dependency_versions.'antlr4-python3-runtime' -eq \"4.13.2\"" in text


def test_math500_wrapper_is_calibration_only_and_non_overwriting():
    text = Path("scripts/math500_gate_windows.ps1").read_text(encoding="utf-8")

    assert "data/processed/math500/calibration.jsonl" in text
    assert "--limit 25" in text
    assert "--count 25" in text
    assert "Require-No-PilotArtifacts" in text
    assert "will not overwrite evidence" in text
    assert "results/math500_scorer_manual_review.json" in text


def test_math500_wrapper_preserves_manual_gate():
    text = Path("scripts/math500_gate_windows.ps1").read_text(encoding="utf-8")

    assert "AUTOMATED MATH-500 SCORER PILOT COMPLETE." in text
    assert "Manual gate remains." in text
    assert "record_math500_scorer_review.py" in text
