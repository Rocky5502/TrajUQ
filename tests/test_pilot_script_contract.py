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
    assert "results/gated_dataset_access.json" in text
    assert "trajuq_gated_dataset_access_v1" in text
    assert "revision_ledger_sha256" in text


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
    assert "scripts/verify_gated_dataset_access.py" in text
    assert "results/gated_dataset_access.json" in text
    assert "trajuq_gated_dataset_access_v1" in text
    assert "scripts/export_research_handoff.py" in text
    assert "dist/trajuq_preprimary_handoff.zip" in text
    assert "--skip-status-refresh" in text


def test_linux_preprimary_freeze_checks_gated_dataset_access():
    text = Path("scripts/freeze_preprimary.sh").read_text(encoding="utf-8")

    assert "scripts/verify_gated_dataset_access.py" in text
    assert "results/gated_dataset_access.json" in text
    assert "trajuq_gated_dataset_access_v1" in text
    assert "revision_ledger_sha256" in text
    assert "PRE-PRIMARY FREEZE PASS" in text


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
    assert "source_git_commit" in text
    assert "source_git_clean" in text
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


def test_human_pilot_gate_requires_parity_and_pilot_run_audits():
    text = Path("scripts/record_manual_gate.py").read_text(encoding="utf-8")
    assert "trajuq_run_audit_v1" in text
    assert "parity_audit_ok" in text
    assert "pilot raw audit is missing, stale, or not PASS" in text
    assert "pilot scored audit is missing, stale, or not PASS" in text
    assert "LLaDA parity instrumented run audit is missing, stale, or not PASS" in text


def test_linux_a100_pilot_wrapper_matches_frozen_contract():
    text = Path("scripts/pilot_linux.sh").read_text(encoding="utf-8")

    assert 'STAGE="${1:-all}"' in text
    assert "results/revision_ledger.json" in text
    assert "results/cpu_verification.json" in text
    assert "trajuq_revision_ledger_v2" in text
    assert "trajuq_cpu_verification_v1" in text
    assert "git status --porcelain" in text
    assert "results/frozen_models/llada.yaml" in text
    assert "scripts/prepare_eval_pair.py" in text
    assert "scripts/preflight.py" in text
    assert "scripts/verify_a100_machine.py" in text
    assert "--device cuda" in text
    assert "--dtype bfloat16" in text
    assert "scripts/freeze_manifest.py" in text
    assert "scripts/run_llada_reference.py" in text
    assert "scripts/run_generation.py" in text
    assert "scripts/compare_run_parity.py" in text
    assert "--require-token-ids" in text
    assert "--require-provenance" in text
    assert "trajuq_exact_parity_v3" in text
    assert "scripts/pilot_quality_gate.py" in text
    assert "trajuq_pilot_quality_gate_v2" in text
    assert "scripts/export_trace_review.py" in text
    assert "source_git_commit" in text
    assert "source_git_clean" in text
    assert "will not delete or overwrite evidence" in text
    assert '"$PARITY_RESULT"' in text
    assert '"$PILOT_GATE"' in text
    assert "rm -f" not in text


def test_linux_remaining_parity_wrapper_is_guarded():
    text = Path("scripts/parity_remaining_linux.sh").read_text(encoding="utf-8")

    assert "results/cpu_verification.json" in text
    assert "results/revision_ledger.json" in text
    assert "git status --porcelain" in text
    assert "scripts/run_sdar_reference.py" in text
    assert "scripts/run_dream_reference.py" in text
    assert "--device cuda" in text
    assert "--dtype bfloat16" in text
    assert "scripts/compare_run_parity.py" in text
    assert "--require-token-ids" in text
    assert "--require-provenance" in text
    assert "trajuq_exact_parity_v3" in text
    assert "shared_records" in text
    assert "mismatch_count" in text
    assert "provenance_mismatch_count" in text
    assert "will not delete or overwrite evidence" in text
    assert '"$parity"' in text
    assert "rm -f" not in text


def test_linux_dream_parity_refuses_existing_parity_artifact():
    text = Path("scripts/parity_remaining_linux.sh").read_text(encoding="utf-8")

    dream_block = text[text.index("run_dream() {"):]
    assert 'require_no_existing_artifacts "$ref" "$inst" "$inst.audit.json" "$parity" "$overhead"' in dream_block


def test_linux_math500_wrapper_is_guarded():
    text = Path("scripts/math500_gate_linux.sh").read_text(encoding="utf-8")

    assert "results/cpu_verification.json" in text
    assert "results/revision_ledger.json" in text
    assert "results/llada_parity.json" in text
    assert "trajuq_math500_scorer_audit_v2" in text
    assert "math-verify" in text
    assert "0.9.0" in text
    assert "antlr4-python3-runtime" in text
    assert "4.13.2" in text
    assert "scripts/prepare_eval_pair.py" in text
    assert "--device cuda" in text
    assert "--dtype bfloat16" in text
    assert "scripts/export_math500_scorer_review.py" in text
    assert "review %s" in text
    assert "will not overwrite evidence" in text
    assert "rm -f" not in text


def test_a100_campaign_runner_preserves_human_gates_and_frozen_order():
    text = Path("scripts/a100_campaign_linux.sh").read_text(encoding="utf-8")

    assert "require_clean_frozen_checkout" in text
    assert "trajuq_revision_ledger_v2" in text
    assert "results/gated_dataset_access.json" in text
    assert "trajuq_gated_dataset_access_v1" in text
    assert "revision_ledger_sha256" in text
    assert "trajuq_primary_readiness_v2" in text
    assert "scripts/pilot_linux.sh prepare" in text
    assert "scripts/pilot_linux.sh parity" in text
    assert "scripts/pilot_linux.sh pilot" in text
    assert "STOP FOR HUMAN REVIEW" in text
    assert "trajuq_pilot_manual_gate_v2" in text
    assert "artifact commit" in text
    assert "artifact was not recorded from a clean checkout" in text
    assert "scripts/parity_remaining_linux.sh all" in text
    assert "scripts/math500_gate_linux.sh prepare" in text
    assert "scripts/math500_gate_linux.sh pilot" in text
    assert "STOP FOR HUMAN MATH-500 REVIEW" in text
    assert "scripts/plan_primary_matrix.py" in text
    assert "scripts/estimate_a100_campaign.py" in text
    assert "estimate)" in text
    assert "scripts/execute_primary_plan.py --execute" in text
    assert "scripts/plan_primary_analysis.py" in text
    assert "scripts/execute_primary_analysis.py --execute" in text
    assert "scripts/plan_repeated_sampling.py" in text
    assert "scripts/execute_repeated_sampling.py --execute" in text
    assert "scripts/plan_adaptive_runs.py" in text
    assert "scripts/execute_adaptive_plan.py --execute" in text
    assert "scripts/render_results_tex.py" in text
    assert "scripts/render_paper_result_figures.py" in text
    assert "all-after-readiness" in text


def test_wrappers_use_shared_frozen_preprimary_validator():
    text = Path("scripts/pilot_windows.ps1").read_text(encoding="utf-8")
    assert "scripts/validate_frozen_preprimary.py" in text
    text = Path("scripts/pilot_linux.sh").read_text(encoding="utf-8")
    assert "scripts/validate_frozen_preprimary.py" in text


def test_llada_parity_reuse_requires_source_bound_run_provenance():
    windows = Path("scripts/pilot_windows.ps1").read_text(encoding="utf-8")
    linux = Path("scripts/pilot_linux.sh").read_text(encoding="utf-8")

    assert "$Parity.run_provenance.source_git_commit -eq $Commit" in windows
    assert "$Parity.run_provenance.source_git_clean -eq $true" in windows
    assert (
        'p.get("run_provenance", {}).get("source_git_commit") == "$FROZEN_COMMIT"'
        in linux
    )
    assert (
        'p.get("run_provenance", {}).get("source_git_clean") is True'
        in linux
    )
