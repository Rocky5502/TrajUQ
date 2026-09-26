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
    assert "scripts/export_research_handoff.py" in text
    assert "dist/trajuq_preprimary_handoff.zip" in text
    assert "--skip-status-refresh" in text
