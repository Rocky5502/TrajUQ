from pathlib import Path


def test_a100_campaign_exports_progress_handoffs_at_expensive_milestones():
    text = Path("scripts/a100_campaign_linux.sh").read_text(encoding="utf-8")

    assert "export_progress_handoff()" in text
    assert "scripts/export_research_handoff.py" in text
    assert "--skip-status-refresh" in text

    for label in (
        "pilot",
        "parity_math",
        "readiness",
        "estimate",
        "primary",
        "analysis",
        "rq2",
        "rq3",
        "paper",
    ):
        assert f'export_progress_handoff "{label}"' in text

    assert 'dist/trajuq_${label}_handoff.zip' in text


def test_a100_campaign_keeps_human_review_stops_after_checkpoint_export():
    text = Path("scripts/a100_campaign_linux.sh").read_text(encoding="utf-8")

    pilot_export = text.index('export_progress_handoff "pilot"')
    pilot_stop = text.index("[TrajUQ] STOP FOR HUMAN REVIEW.")
    assert pilot_export < pilot_stop

    math_export = text.index('export_progress_handoff "parity_math"')
    math_stop = text.index("[TrajUQ] STOP FOR HUMAN MATH-500 REVIEW.")
    assert math_export < math_stop
