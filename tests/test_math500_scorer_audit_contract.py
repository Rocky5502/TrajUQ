from scripts.audit_math500_scorer import (
    EXPECTED_ANTLR_VERSION,
    EXPECTED_MATH_VERIFY_VERSION,
    dependency_failures,
)


def test_dependency_failures_accepts_frozen_versions():
    observed = {
        "math-verify": EXPECTED_MATH_VERIFY_VERSION,
        "antlr4-python3-runtime": EXPECTED_ANTLR_VERSION,
    }
    assert dependency_failures(observed) == []


def test_dependency_failures_rejects_version_drift():
    failures = dependency_failures(
        {
            "math-verify": "0.8.0",
            "antlr4-python3-runtime": None,
        }
    )

    assert {item["package"] for item in failures} == {
        "math-verify",
        "antlr4-python3-runtime",
    }


def test_scorer_audit_records_clean_checkout_contract():
    from pathlib import Path

    text = Path("scripts/audit_math500_scorer.py").read_text(encoding="utf-8")
    assert "def git_clean()" in text
    assert "git_worktree_not_clean" in text
    assert '"git_clean": clean' in text
