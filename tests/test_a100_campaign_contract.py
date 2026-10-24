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


def test_a100_parity_math_revalidates_source_bound_human_pilot_gate():
    text = Path("scripts/a100_campaign_linux.sh").read_text(encoding="utf-8")

    assert "require_pilot_manual_gate()" in text
    assert "review_sheet_sha256" in text
    assert "parity_artifact_sha256" in text
    assert "pilot_quality_sha256" in text
    assert "review_scored_sha256" in text
    assert "trajuq_run_audit_v1" in text
    assert "reference_sha256" in text
    assert "instrumented_sha256" in text
    assert 'manual.get("git_commit") != head' in text

    parity_math = text[text.index("  parity-math)"):]
    assert "require_pilot_manual_gate" in parity_math
    assert (
        'require_json_pass       "results/llada_gsm8k_pilot50_manual_review.json"'
        not in parity_math
    )


def test_a100_campaign_revalidates_primary_readiness_sources():
    text = Path("scripts/a100_campaign_linux.sh").read_text(encoding="utf-8")

    block_start = text.index("require_primary_readiness()")
    block_end = text.index("require_pilot_manual_gate()")
    block = text[block_start:block_end]

    assert "scripts/primary_readiness.py" in block
    assert "results/primary_readiness.json" in block
    assert "trajuq_primary_readiness_v2" in block


def test_generic_campaign_gate_accepts_experiment_commit_field():
    text = Path("scripts/a100_campaign_linux.sh").read_text(encoding="utf-8")

    assert 'payload.get("experiment_git_commit")' in text
    assert "artifact commit" in text


def test_a100_rq2_uses_plan_level_resident_generation_before_canonical_executor():
    text = Path("scripts/a100_campaign_linux.sh").read_text(encoding="utf-8")

    assert "scripts/plan_repeated_sampling.py" in text
    assert "scripts/resident_generate_repeated_plan.py" in text
    assert "scripts/execute_repeated_sampling.py --execute" in text
    assert text.index("scripts/resident_generate_repeated_plan.py") < text.index(
        "scripts/execute_repeated_sampling.py --execute"
    )
