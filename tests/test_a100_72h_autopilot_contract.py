from pathlib import Path


def test_72h_autopilot_requires_readiness_and_prioritizes_core_stages():
    text = Path("scripts/a100_72h_autopilot.sh").read_text(encoding="utf-8")

    assert "bash scripts/a100_campaign_linux.sh readiness" in text
    assert 'BUDGET_HOURS="\${BUDGET_HOURS:-72}"' in text
    assert 'MONITOR_SECONDS="\${MONITOR_SECONDS:-3600}"' in text
    assert "timeout --foreground --signal=TERM" in text
    assert "nvidia-smi" in text
    assert "hourly_monitor.log" in text
    assert "a100_72h_autopilot_state.json" in text
    assert "for stage in estimate primary analysis rq3 rq2 paper" in text


def test_72h_autopilot_never_runs_human_gate_recorders():
    text = Path("scripts/a100_72h_autopilot.sh").read_text(encoding="utf-8")

    assert "record_manual_gate.py" not in text
    assert "record_math500_scorer_review.py" not in text
    assert "parity-math" not in text
    assert "pilot)" not in text


def test_campaign_exposes_autopilot_stage():
    text = Path("scripts/a100_campaign_linux.sh").read_text(encoding="utf-8")

    assert "autopilot-72h)" in text
    assert "bash scripts/a100_72h_autopilot.sh" in text


def test_72h_autopilot_preserves_failed_stage_exit_code():
    text = Path("scripts/a100_72h_autopilot.sh").read_text(encoding="utf-8")

    assert 'if ! run_stage "$stage"' not in text
    assert 'run_stage "$stage"' in text
    assert 'rc=$?' in text
    assert 'if (( rc != 0 )); then' in text
    assert 'exit "$rc"' in text



def test_72h_autopilot_inherits_single_paid_rental_deadline():
    text = Path("scripts/a100_72h_autopilot.sh").read_text(encoding="utf-8")

    assert "results/a100_rental_clock.json" in text
    assert "trajuq_a100_rental_clock_v1" in text
    assert "Rental clock commit differs from current frozen HEAD" in text
    assert "BUDGET_HOURS=" in text
    assert 'START_EPOCH="$(date +%s)"' not in text
    assert "check_estimate_against_remaining_budget" in text
    assert "BUDGET_FORECAST_FAIL" in text
    assert "mean projection exceeds the remaining paid-rental" in text


def test_campaign_exposes_rental_start_stage():
    text = Path("scripts/a100_campaign_linux.sh").read_text(encoding="utf-8")

    assert "rental-start)" in text
    assert "scripts/start_a100_rental_clock.py" in text
