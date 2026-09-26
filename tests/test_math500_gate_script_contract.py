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
    assert "$Audit.git_clean -eq $true" in text


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


def test_math500_wrapper_revalidates_hash_bound_split_and_parity_sources():
    text = Path("scripts/math500_gate_windows.ps1").read_text(encoding="utf-8")

    assert "function Get-FileSha256" in text
    assert "reference_sha256" in text
    assert "instrumented_sha256" in text
    assert "$Parity.git_clean -eq $true" in text
    assert "function Test-RunAuditPass" in text
    assert '$InstrumentedAuditOk = Test-RunAuditPass $Instrumented "raw"' in text
    assert "$InstrumentedAuditOk" in text
    assert "function Assert-Math500FrozenPair" in text
    assert "trajuq_split_pair_audit_v1" in text
    assert "calibration_sha256" in text
    assert "test_sha256" in text
    assert "$Audit.requested_revision -ne $DatasetSha" in text
    assert text.count("Assert-Math500FrozenPair") == 3


def test_math500_human_review_is_calibration_source_bound():
    record_text = Path("scripts/record_math500_scorer_review.py").read_text(encoding="utf-8")
    readiness_text = Path("scripts/primary_readiness.py").read_text(encoding="utf-8")

    assert "review_scored_artifact" in record_text
    assert "review_scored_sha256" in record_text
    assert "review_input" in record_text
    assert "review_input_sha256" in record_text
    assert "data/processed/math500/calibration.jsonl" in record_text
    assert "math_scored_audit_ok" in readiness_text
    assert "math_input_hash_ok" in readiness_text
    assert "reviewed MATH-500 scored artifact has no current PASS audit" in record_text
    assert "review_scored_audit_ok" in record_text
