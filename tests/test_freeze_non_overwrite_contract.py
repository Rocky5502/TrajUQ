from pathlib import Path


def test_freeze_producers_refuse_overwrite_by_default():
    resolver = Path("scripts/resolve_experiment_revisions.py").read_text(
        encoding="utf-8"
    )
    cpu = Path("scripts/verify_cpu.py").read_text(encoding="utf-8")
    gated = Path("scripts/verify_gated_dataset_access.py").read_text(
        encoding="utf-8"
    )
    contracts = Path("scripts/verify_dataset_contracts.py").read_text(
        encoding="utf-8"
    )
    math_audit = Path("scripts/audit_math500_scorer.py").read_text(
        encoding="utf-8"
    )
    cpu_verifier = Path("scripts/verify_cpu.py").read_text(encoding="utf-8")

    assert '--overwrite' in resolver
    assert "Refusing to overwrite frozen revision evidence" in resolver
    assert '--overwrite' in cpu
    assert "Refusing to overwrite CPU verification evidence" in cpu
    assert '--overwrite' in gated
    assert "Refusing to overwrite gated dataset access evidence" in gated
    assert '--overwrite' in contracts
    assert "Refusing to overwrite dataset-contract evidence" in contracts
    assert '--overwrite' in math_audit
    assert "Refusing to overwrite MATH-500 scorer audit evidence" in math_audit
    assert 'math500_audit_argv.append("--overwrite")' in cpu_verifier


def test_official_freeze_wrappers_never_request_overwrite():
    windows = Path("scripts/freeze_preprimary_windows.ps1").read_text(
        encoding="utf-8"
    )
    linux = Path("scripts/freeze_preprimary.sh").read_text(encoding="utf-8")

    assert "--overwrite" not in windows
    assert "--overwrite" not in linux


def test_revision_resolver_guards_ledger_and_frozen_configs_together():
    text = Path("scripts/resolve_experiment_revisions.py").read_text(
        encoding="utf-8"
    )

    assert "frozen_targets" in text
    assert "results/revision_ledger.json" in text
    assert "Official freeze" in text


def test_official_freeze_wrappers_preflight_all_core_evidence_before_running():
    windows = Path("scripts/freeze_preprimary_windows.ps1").read_text(
        encoding="utf-8"
    )
    linux = Path("scripts/freeze_preprimary.sh").read_text(encoding="utf-8")

    for text in (windows, linux):
        assert "results/cpu_verification.json" in text
        assert "results/revision_ledger.json" in text
        assert "results/gated_dataset_access.json" in text
        assert "results/dataset_contracts.json" in text
        assert "results/math500_scorer_audit.json" in text
        assert "results/cpu_verification_logs" in text
        assert "results/frozen_models" in text
        assert "Refusing to start a mixed pre-primary freeze" in text
