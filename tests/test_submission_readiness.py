import hashlib
import json

import scripts.submission_readiness as submission_readiness
from scripts.submission_readiness import (
    verify_downstream_provenance,
    verify_generated_results,
)


def test_generated_results_hash_gate_passes(tmp_path):
    source = tmp_path / "macro.json"
    source.write_text('{"ok": true}', encoding="utf-8")
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
                "sources": {
                    "primary": {
                        "path": str(source),
                        "sha256": hashlib.sha256(
                            source.read_bytes()
                        ).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = verify_generated_results(manifest, generated)
    assert result["status"] == "PASS"


def test_generated_results_hash_gate_rejects_changed_source(tmp_path):
    source = tmp_path / "macro.json"
    source.write_text('{"ok": true}', encoding="utf-8")
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
                "sources": {
                    "primary": {
                        "path": str(source),
                        "sha256": hashlib.sha256(
                            source.read_bytes()
                        ).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
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
