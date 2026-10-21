from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import finalize_rq2_paper


def test_finalizer_runs_exact_promotion_chain(monkeypatch, tmp_path, capsys) -> None:
    experiment_root = tmp_path / "experiment"
    paper_root = tmp_path / "paper"
    experiment_root.mkdir()
    paper_root.mkdir()
    plan_path = Path("results/repeated_sampling_plan.json")
    calls: list[tuple[list[str], Path, bool]] = []

    monkeypatch.setattr(
        finalize_rq2_paper,
        "git",
        lambda root, *args: (
            finalize_rq2_paper.EXPECTED_COMMIT
            if args == ("rev-parse", "HEAD")
            else ""
        ),
    )
    monkeypatch.setattr(
        finalize_rq2_paper,
        "require_generation_complete",
        lambda root, plan: {"status": "PASS"},
    )
    monkeypatch.setattr(
        finalize_rq2_paper,
        "run",
        lambda argv, cwd, dry_run: calls.append((argv, cwd, dry_run)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_rq2_paper.py",
            "--experiment-root",
            str(experiment_root),
            "--paper-root",
            str(paper_root),
            "--plan",
            plan_path.as_posix(),
            "--dry-run",
        ],
    )

    assert finalize_rq2_paper.main() == 0
    assert calls == [
        (
            [
                sys.executable,
                "scripts/execute_repeated_sampling.py",
                "--plan",
                plan_path.as_posix(),
                "--execute",
            ],
            experiment_root.resolve(),
            True,
        ),
        (
            [
                sys.executable,
                str(paper_root.resolve() / "scripts/build_rq2_final_evidence.py"),
                "--root",
                str(experiment_root.resolve()),
                "--plan",
                plan_path.as_posix(),
            ],
            paper_root.resolve(),
            True,
        ),
        (
            [sys.executable, "scripts/paper_evidence_pipeline.py", "final"],
            paper_root.resolve(),
            True,
        ),
    ]
    assert json.loads(capsys.readouterr().out) == {"dry_run": True, "status": "PASS"}


def test_finalizer_fails_closed_before_commands_when_generation_is_incomplete(
    monkeypatch, tmp_path, capsys
) -> None:
    experiment_root = tmp_path / "experiment"
    paper_root = tmp_path / "paper"
    experiment_root.mkdir()
    paper_root.mkdir()
    calls: list[list[str]] = []

    monkeypatch.setattr(
        finalize_rq2_paper,
        "git",
        lambda root, *args: (
            finalize_rq2_paper.EXPECTED_COMMIT
            if args == ("rev-parse", "HEAD")
            else ""
        ),
    )

    def incomplete(root: Path, plan: Path) -> dict:
        raise ValueError("generation is incomplete")

    monkeypatch.setattr(finalize_rq2_paper, "require_generation_complete", incomplete)
    monkeypatch.setattr(
        finalize_rq2_paper,
        "run",
        lambda argv, cwd, dry_run: calls.append(argv),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_rq2_paper.py",
            "--experiment-root",
            str(experiment_root),
            "--paper-root",
            str(paper_root),
        ],
    )

    assert finalize_rq2_paper.main() == 2
    assert calls == []
    assert json.loads(capsys.readouterr().out) == {
        "reason": "generation is incomplete",
        "status": "BLOCKED",
    }
