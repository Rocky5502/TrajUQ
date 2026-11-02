from pathlib import Path


def test_rq5_renderer_is_claim_gated_and_artifact_driven():
    text = Path("scripts/render_authorized_rq5_tex.py").read_text(encoding="utf-8")
    assert 'claim.get("claim_authorized") is not True' in text
    assert "RQ5 manuscript binding drift" in text
    assert "results/rq5/analysis/macro_aggregate.json" in text
    assert "results/rq5/analysis/validation_summary.json" in text
    assert "did not improve" in text


def test_manuscript_invokes_authorized_rq5_interface():
    text = Path("paper/main.tex").read_text(encoding="utf-8")
    assert "\\RQFiveGeneratedResults" in text
