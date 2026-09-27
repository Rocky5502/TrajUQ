from pathlib import Path

import pytest

from scripts.start_a100_rental_clock import frozen_target_hours


def test_frozen_target_hours_reads_declared_budget(tmp_path: Path):
    cfg = tmp_path / "primary.yaml"
    cfg.write_text(
        "compute_budget:\n  target_wall_clock_hours: 72\n",
        encoding="utf-8",
    )
    assert frozen_target_hours(cfg) == 72


def test_frozen_target_hours_rejects_missing_or_nonpositive_budget(tmp_path: Path):
    missing = tmp_path / "missing.yaml"
    missing.write_text("compute_budget: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot resolve"):
        frozen_target_hours(missing)

    zero = tmp_path / "zero.yaml"
    zero.write_text(
        "compute_budget:\n  target_wall_clock_hours: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be positive"):
        frozen_target_hours(zero)


def test_rental_clock_script_is_immutable_and_commit_bound():
    text = Path("scripts/start_a100_rental_clock.py").read_text(
        encoding="utf-8"
    )
    assert "trajuq_a100_rental_clock_v1" in text
    assert 'git_text("status", "--porcelain")' in text
    assert "git_commit" in text
    assert "Existing rental clock" in text
    assert "do not silently restart time" in text
