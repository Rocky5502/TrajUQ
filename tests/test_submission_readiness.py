import hashlib
import json

from scripts.submission_readiness import verify_generated_results


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
