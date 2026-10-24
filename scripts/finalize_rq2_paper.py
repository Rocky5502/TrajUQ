#!/usr/bin/env python
"""Fail-closed one-command RQ2 finalization and paper promotion.

This command refuses to start while any frozen generation partition is
incomplete.  Once generation is complete, the repository's resumable RQ2
executor performs scoring, comparisons, and aggregation; the final manifest is
then built and the all-RQ paper evidence gate is invoked.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "28002055c828c6910840de6f2105aa6367514144"


@contextmanager
def working_directory(path: Path):
    import os
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT).strip()


def require_generation_complete(root: Path, plan_path: Path) -> dict:
    sys.path.insert(0, str(root))
    from scripts.execute_repeated_sampling import generation_state, validate_plan_provenance

    plan = json.loads((root / plan_path).read_text(encoding="utf-8"))
    plan["_plan_path"] = plan_path.as_posix()
    if plan.get("status") != "PASS" or plan.get("experiment_git_commit") != EXPECTED_COMMIT:
        raise ValueError("RQ2 plan is not a PASS at the authoritative commit")
    with working_directory(root):
        failures = validate_plan_provenance(plan)
        if failures:
            raise ValueError("RQ2 plan provenance failed: " + "; ".join(failures))
        incomplete: list[str] = []
        for cell in plan.get("cells", []):
            seeds = [int(seed) for seed in cell["seeds"]]
            for partition, key in (("calibration", "calibration_generation_argv"), ("test", "test_generation_argv")):
                state = generation_state([str(value) for value in cell[key]], seeds, EXPECTED_COMMIT)
                if not state["complete"]:
                    incomplete.append(f"{cell['cell']}:{partition}")
        if incomplete:
            raise ValueError(
                "generation is incomplete; GPU execution remains authoritative and untouched: "
                + ", ".join(incomplete)
            )
    return plan


def run(argv: list[str], cwd: Path, dry_run: bool) -> None:
    print("[DRY]" if dry_run else "[RUN]", " ".join(argv))
    if not dry_run:
        subprocess.run(argv, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", default=".")
    parser.add_argument("--paper-root", default=str(PAPER_ROOT))
    parser.add_argument("--plan", default="results/repeated_sampling_plan.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    experiment_root = Path(args.experiment_root).resolve()
    paper_root = Path(args.paper_root).resolve()
    plan_path = Path(args.plan)
    try:
        if git(experiment_root, "rev-parse", "HEAD") != EXPECTED_COMMIT:
            raise ValueError("experiment checkout is not the authoritative commit")
        if git(experiment_root, "status", "--porcelain"):
            raise ValueError("experiment checkout is dirty")
        require_generation_complete(experiment_root, plan_path)
        run([
            sys.executable, "scripts/execute_repeated_sampling.py", "--plan", plan_path.as_posix(), "--execute"
        ], experiment_root, args.dry_run)
        run([
            sys.executable, str(paper_root / "scripts/build_rq2_final_evidence.py"),
            "--root", str(experiment_root), "--plan", plan_path.as_posix(),
        ], paper_root, args.dry_run)
        run([
            sys.executable, "scripts/paper_evidence_pipeline.py", "final"
        ], paper_root, args.dry_run)
    except (ImportError, KeyError, OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "PASS", "dry_run": args.dry_run}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
