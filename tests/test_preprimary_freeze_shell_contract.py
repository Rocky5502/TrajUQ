from pathlib import Path


def test_shell_preprimary_freeze_matches_windows_integrity_contract():
    text = Path("scripts/freeze_preprimary.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert "git status --porcelain" in text
    assert "git rev-parse HEAD" in text
    assert '".[dev,math,models]"' in text
    assert "scripts/verify_cpu.py" in text
    assert "trajuq_cpu_verification_v1" in text
    assert "scripts/resolve_experiment_revisions.py" in text
    assert "trajuq_revision_ledger_v2" in text
    assert "Tracked worktree changed during the pre-primary freeze" in text


def test_shell_preprimary_freeze_exports_private_handoff():
    text = Path("scripts/freeze_preprimary.sh").read_text(encoding="utf-8")

    assert "scripts/project_status.py" in text
    assert "scripts/export_research_handoff.py" in text
    assert "dist/trajuq_preprimary_handoff.zip" in text
    assert "--skip-status-refresh" in text
    assert "PRE-PRIMARY FREEZE PASS" in text
