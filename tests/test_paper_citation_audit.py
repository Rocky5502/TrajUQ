from scripts.audit_paper import main as audit_main


def test_paper_audit_module_imports():
    # Full pre-results audit is exercised by tests/test_paper_audit.py.
    # This import guard catches syntax regressions in the static checker itself.
    assert callable(audit_main)
