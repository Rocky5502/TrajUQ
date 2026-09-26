from pathlib import Path


def test_windows_pilot_wrapper_has_single_stage_blocks():
    text = Path("scripts/pilot_windows.ps1").read_text(encoding="utf-8")
    assert text.count('if ($Stage -in @("prepare", "all"))') == 1
    assert text.count('if ($Stage -in @("parity", "all"))') == 1
    assert text.count('if ($Stage -in @("pilot", "all"))') == 1
    assert "results/frozen_models/llada.yaml" in text
    assert "results/llada_pilot_model_config.yaml" not in text
    assert "compare_run_parity.py" in text
    assert "pilot_quality_gate.py" in text
    assert "git status --porcelain" in text
    assert "results/cpu_verification.json" in text
    assert "trajuq_cpu_verification_v1" in text


def test_windows_pilot_wrapper_has_no_known_corruption_fragments():
    text = Path("scripts/pilot_windows.ps1").read_text(encoding="utf-8")
    assert text.count("AUTOMATED PILOT COMPLETE.") == 1
    assert text.count("Prepare frozen GSM8K pair") == 1
    assert text.count("20-prompt native reference") == 1
    assert text.count("50-item TrajUQ pilot") == 1


def test_local_cpu_verifier_matches_ci_dependency_scope():
    text = Path("scripts/verify_cpu.ps1").read_text(encoding="utf-8")
    assert '".[dev,math]"' in text
    assert "audit_paper.py" in text


def test_preprimary_freeze_exports_private_handoff():
    text = Path("scripts/freeze_preprimary_windows.ps1").read_text(encoding="utf-8")
    assert "trajuq_cpu_verification_v1" in text
    assert "$Cpu.git_clean -ne $true" in text
    assert "$DirtyAfterFreeze" in text
    assert "scripts/export_research_handoff.py" in text
    assert "dist/trajuq_preprimary_handoff.zip" in text
    assert "--skip-status-refresh" in text


def test_windows_pilot_wrapper_is_non_overwriting_and_hash_bound():
    text = Path("scripts/pilot_windows.ps1").read_text(encoding="utf-8")

    assert "Remove-Item" not in text
    assert "Remove-Artifact" not in text
    assert "Require-No-ExistingArtifacts" in text
    assert "will not delete or overwrite evidence" in text
    assert "reference_sha256" in text
    assert "instrumented_sha256" in text
    assert "raw_sha256" in text
    assert "scored_sha256" in text
    assert "calibration_sha256" in text
    assert "test_sha256" in text
    assert "$Parity.git_clean -eq $true" in text
    assert "$Gate.git_clean -eq $true" in text
    assert "function Test-RunAuditPass" in text
    assert '$RawAuditOk = Test-RunAuditPass $Raw "raw"' in text
    assert '$ScoredAuditOk = Test-RunAuditPass $Scored "scored"' in text
    assert "$RawAuditOk -and" in text
    assert "$ScoredAuditOk -and" in text
    assert '$InstrumentedAuditOk = Test-RunAuditPass $Instrumented "raw"' in text
    assert "$InstrumentedAuditOk" in text


def test_gpu_stages_revalidate_frozen_pilot_inputs():
    text = Path("scripts/pilot_windows.ps1").read_text(encoding="utf-8")
    assert "function Assert-FrozenPilotInputs" in text
    assert text.count("Assert-FrozenPilotInputs") == 2
    stage = text.index('if ($Stage -in @("parity", "pilot", "all"))')
    parity = text.index('if ($Stage -in @("parity", "all"))')
    between = text[stage:parity]
    assert "Assert-FrozenPilotInputs" in between
    assert "-FreezeManifestPath \"results/pilot_freeze_manifest.json\"" in between
    assert "-DatasetSha $DatasetSha" in between
    assert "-Commit $Ledger.git_commit" in between


def test_pilot_human_review_is_source_bound():
    export_text = Path("scripts/export_trace_review.py").read_text(encoding="utf-8")
    record_text = Path("scripts/record_manual_gate.py").read_text(encoding="utf-8")
    readiness_text = Path("scripts/primary_readiness.py").read_text(encoding="utf-8")

    assert "- scored artifact: " in export_text
    assert "- scored SHA-256: " in export_text
    assert "review_scored_artifact" in record_text
    assert "review_scored_sha256" in record_text
    assert "LLaDA parity {label} source hash is stale" in record_text
    assert "review_source_ok" in readiness_text
    assert "pilot raw audit is missing, stale, or not PASS" in record_text
    assert "pilot scored audit is missing, stale, or not PASS" in record_text
    assert "pilot_raw_audit_ok" in record_text
    assert "pilot_scored_audit_ok" in record_text
