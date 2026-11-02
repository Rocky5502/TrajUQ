from pathlib import Path


def test_systems_audit_is_non_claim_authorizing_and_preserves_discrepancy():
    text = Path("scripts/audit_systems_diagnostics.py").read_text(encoding="utf-8")
    assert '"scientific_claim_authority": False' in text
    assert '"confirmatory_rq1_rq5": False' in text
    assert "PASS_WITH_PRESERVED_STALE_INTERNAL_PROGRESS_HASH" in text
    assert '"raw_artifacts_modified": False' in text
    assert 'with destination.open("xb")' in text
