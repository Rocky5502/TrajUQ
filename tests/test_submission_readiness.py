import hashlib
import json

import scripts.submission_readiness as submission_readiness
from scripts.submission_readiness import (
    preprimary_revalidation_argv,
    verify_downstream_provenance,
    verify_generated_results,
    verify_submission_checkout,
)


def make_render_sources(tmp_path):
    sources = {}
    for name in ("primary", "claim_gate", "repeated", "adaptive"):
        path = tmp_path / f"{name}.json"
        path.write_text('{"ok": true}', encoding="utf-8")
        sources[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    config = tmp_path / "primary.yaml"
    config.write_text("experiment: {}\n", encoding="utf-8")
    sources["primary_config"] = {
        "path": str(config),
        "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    return sources


def test_generated_results_hash_gate_passes(tmp_path):
    generated = tmp_path / "generated_results.tex"
    generated.write_text(
        "\\newcommand{\\PrimaryTrajAUROC}{0.700}\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "trajuq_paper_results_render_v1",
                "output_sha256": hashlib.sha256(
                    generated.read_bytes()
                ).hexdigest(),
                "sources": make_render_sources(tmp_path),
                "macro_count": 1,
                "rq1_claim_decision": "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM",
            }
        ),
        encoding="utf-8",
    )

    result = verify_generated_results(manifest, generated)
    assert result["status"] == "PASS"


def test_generated_results_hash_gate_rejects_changed_source(tmp_path):
    generated = tmp_path / "generated_results.tex"
    generated.write_text(
        "\\newcommand{\\PrimaryTrajAUROC}{0.700}\n",
        encoding="utf-8",
    )
    sources = make_render_sources(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "trajuq_paper_results_render_v1",
                "output_sha256": hashlib.sha256(
                    generated.read_bytes()
                ).hexdigest(),
                "sources": sources,
                "macro_count": 1,
                "rq1_claim_decision": "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM",
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "primary.json"
    source.write_text('{"ok": false}', encoding="utf-8")

    result = verify_generated_results(manifest, generated)
    assert result["status"] == "FAIL"
    assert any("source hash changed" in item for item in result["failures"])


def test_generated_results_gate_rejects_pre_results_placeholder(tmp_path):
    generated = tmp_path / "generated_results.tex"
    generated.write_text(
        "% PRE-RESULTS FALLBACK\n\\newcommand{\\X}{\\tbd}\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "trajuq_paper_results_render_v1",
                "output_sha256": hashlib.sha256(
                    generated.read_bytes()
                ).hexdigest(),
                "sources": {},
            }
        ),
        encoding="utf-8",
    )
    result = verify_generated_results(manifest, generated)
    assert result["status"] == "FAIL"


def test_downstream_provenance_revalidation_uses_historical_mode(
    tmp_path,
    monkeypatch,
):
    paths = []
    for name in ("primary", "analysis", "repeated", "adaptive"):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "protocol": f"{name}_protocol",
                    "experiment_git_commit": "a" * 40,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    calls = []

    def validator(payload, *, require_current_commit=True):
        calls.append(
            {
                "protocol": payload["protocol"],
                "require_current_commit": require_current_commit,
            }
        )
        return []

    monkeypatch.setattr(
        submission_readiness,
        "validate_primary_provenance",
        validator,
    )
    monkeypatch.setattr(
        submission_readiness,
        "validate_analysis_provenance",
        validator,
    )
    monkeypatch.setattr(
        submission_readiness,
        "validate_repeated_provenance",
        validator,
    )
    monkeypatch.setattr(
        submission_readiness,
        "validate_adaptive_provenance",
        validator,
    )

    result = verify_downstream_provenance(*paths)
    assert result["status"] == "PASS"
    assert len(calls) == 4
    assert all(
        call["require_current_commit"] is False
        for call in calls
    )


def test_downstream_provenance_revalidation_surfaces_drift(
    tmp_path,
    monkeypatch,
):
    paths = []
    for name in ("primary", "analysis", "repeated", "adaptive"):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "protocol": f"{name}_protocol",
                    "experiment_git_commit": "b" * 40,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    monkeypatch.setattr(
        submission_readiness,
        "validate_primary_provenance",
        lambda payload, *, require_current_commit=True: [],
    )
    monkeypatch.setattr(
        submission_readiness,
        "validate_analysis_provenance",
        lambda payload, *, require_current_commit=True: [],
    )
    monkeypatch.setattr(
        submission_readiness,
        "validate_repeated_provenance",
        lambda payload, *, require_current_commit=True: [
            "TrajUQ predictions hash drifted"
        ],
    )
    monkeypatch.setattr(
        submission_readiness,
        "validate_adaptive_provenance",
        lambda payload, *, require_current_commit=True: [],
    )

    result = verify_downstream_provenance(*paths)
    assert result["status"] == "FAIL"
    assert any(
        "repeated: TrajUQ predictions hash drifted" in failure
        for failure in result["failures"]
    )


def test_submission_revalidation_uses_historical_mode_and_separate_output():
    argv = preprimary_revalidation_argv()
    assert "scripts/primary_readiness.py" in argv
    assert "--historical-provenance-only" in argv
    assert "results/submission_preprimary_revalidation.json" in argv
    assert "results/primary_readiness.json" not in argv


def test_submission_checkout_allows_later_clean_commit(monkeypatch):
    def fake_git_text(*args):
        if args == ("rev-parse", "HEAD"):
            return "submission-commit"
        if args == ("status", "--porcelain"):
            return ""
        return None

    monkeypatch.setattr(submission_readiness, "git_text", fake_git_text)
    result = verify_submission_checkout()
    assert result["status"] == "PASS"
    assert result["git_commit"] == "submission-commit"
    assert result["git_clean"] is True


def test_submission_checkout_rejects_dirty_worktree(monkeypatch):
    def fake_git_text(*args):
        if args == ("rev-parse", "HEAD"):
            return "submission-commit"
        if args == ("status", "--porcelain"):
            return " M paper/main.tex"
        return None

    monkeypatch.setattr(submission_readiness, "git_text", fake_git_text)
    result = verify_submission_checkout()
    assert result["status"] == "FAIL"
    assert result["git_clean"] is False
    assert any("worktree is dirty" in item for item in result["failures"])


def test_generated_results_gate_rejects_missing_required_sources(tmp_path):
    generated = tmp_path / "generated_results.tex"
    generated.write_text(
        "\\newcommand{\\PrimaryTrajAUROC}{0.700}\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "trajuq_paper_results_render_v1",
                "output_sha256": hashlib.sha256(
                    generated.read_bytes()
                ).hexdigest(),
                "sources": {},
                "macro_count": 1,
                "rq1_claim_decision": "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM",
            }
        ),
        encoding="utf-8",
    )

    result = verify_generated_results(manifest, generated)
    assert result["status"] == "FAIL"
    assert any("missing required sources" in item for item in result["failures"])
