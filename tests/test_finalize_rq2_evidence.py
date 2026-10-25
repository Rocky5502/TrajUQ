from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import finalize_rq2_evidence


def test_finalizer_runs_registered_science_chain_without_authorizing_claims(
    monkeypatch, tmp_path, capsys
) -> None:
    calls: list[tuple[list[str], Path, bool]] = []
    monkeypatch.setattr(
        finalize_rq2_evidence,
        "git",
        lambda root, *args: (
            finalize_rq2_evidence.EXPECTED_COMMIT
            if args == ("rev-parse", "HEAD")
            else ""
        ),
    )
    monkeypatch.setattr(
        finalize_rq2_evidence, "require_generation_complete", lambda root, plan: None
    )
    monkeypatch.setattr(
        finalize_rq2_evidence,
        "run",
        lambda argv, cwd, dry_run: calls.append((argv, cwd, dry_run)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["finalize_rq2_evidence.py", "--root", str(tmp_path), "--dry-run"],
    )

    assert finalize_rq2_evidence.main() == 0
    assert len(calls) == 2
    assert calls[0][0][1:] == [
        "scripts/execute_repeated_sampling.py",
        "--plan",
        "results/repeated_sampling_plan.json",
        "--state-output",
        "results/repeated_sampling_execution_state.json",
        "--execute",
    ]
    assert calls[1][0][1] == str(
        finalize_rq2_evidence.ROOT / "scripts/build_rq2_final_evidence.py"
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PASS"
    assert result["claim_authorized"] is False


def test_finalizer_fails_before_commands_when_generation_is_incomplete(
    monkeypatch, tmp_path, capsys
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        finalize_rq2_evidence,
        "git",
        lambda root, *args: (
            finalize_rq2_evidence.EXPECTED_COMMIT
            if args == ("rev-parse", "HEAD")
            else ""
        ),
    )

    def incomplete(root: Path, plan: Path) -> None:
        raise ValueError("generation is incomplete")

    monkeypatch.setattr(finalize_rq2_evidence, "require_generation_complete", incomplete)
    monkeypatch.setattr(
        finalize_rq2_evidence,
        "run",
        lambda argv, cwd, dry_run: calls.append(argv),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["finalize_rq2_evidence.py", "--root", str(tmp_path)],
    )

    assert finalize_rq2_evidence.main() == 2
    assert calls == []
    assert json.loads(capsys.readouterr().out) == {
        "reason": "generation is incomplete",
        "status": "BLOCKED",
    }


def test_candidate_requires_exact_seed_and_partition_counts(tmp_path: Path) -> None:
    output = Path("candidate.json")
    payload = {
        "schema": "trajuq_final_rq_evidence_v1",
        "rq": "RQ2",
        "status": "FINAL",
        "experiment_commit": finalize_rq2_evidence.EXPECTED_COMMIT,
        "gates": {
            "seed_jobs_expected": 120,
            "seed_jobs_complete": 119,
            "partitions_expected": 24,
            "partitions_complete": 24,
        },
    }
    (tmp_path / output).write_text(json.dumps(payload), encoding="utf-8")
    try:
        finalize_rq2_evidence.validate_candidate(tmp_path, output)
    except ValueError as exc:
        assert "failed final gate validation" in str(exc)
    else:
        raise AssertionError("119/120 audit gates must fail closed")
