#!/usr/bin/env python
"""Fail-closed, manuscript-free RQ2 evidence finalization.

The command refuses to start until every frozen repeated-sampling generation
partition is complete and audit-bound. It then invokes the repository's
registered scoring/aggregation executor and builds a candidate final evidence
manifest. It never edits ``ARTIFACT_MANIFEST.json`` or authorizes claims.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def require_generation_complete(root: Path, plan_path: Path) -> None:
    sys.path.insert(0, str(root))
    from scripts.execute_repeated_sampling import generation_state, validate_plan_provenance

    plan = json.loads((root / plan_path).read_text(encoding="utf-8"))
    plan["_plan_path"] = plan_path.as_posix()
    if (
        plan.get("protocol") != "trajuq_repeated_sampling_plan_v2"
        or plan.get("status") != "PASS"
        or plan.get("experiment_git_commit") != EXPECTED_COMMIT
        or int(plan.get("expected_cells", -1)) != 12
        or len(plan.get("cells", [])) != 12
    ):
        raise ValueError("RQ2 plan is not the frozen 12-cell PASS protocol")

    with working_directory(root):
        failures = validate_plan_provenance(plan)
        if failures:
            raise ValueError("RQ2 plan provenance failed: " + "; ".join(failures))
        incomplete: list[str] = []
        for cell in plan["cells"]:
            seeds = [int(seed) for seed in cell["seeds"]]
            for partition, key in (
                ("calibration", "calibration_generation_argv"),
                ("test", "test_generation_argv"),
            ):
                state = generation_state(
                    [str(value) for value in cell[key]], seeds, EXPECTED_COMMIT
                )
                if not state["complete"]:
                    incomplete.append(f"{cell['cell']}:{partition}")
        if incomplete:
            raise ValueError(
                "generation is incomplete; canonical execution remains authoritative: "
                + ", ".join(incomplete)
            )


def run(argv: list[str], cwd: Path, dry_run: bool) -> None:
    print("[DRY]" if dry_run else "[RUN]", " ".join(argv))
    if not dry_run:
        subprocess.run(argv, cwd=cwd, check=True)


def validate_candidate(root: Path, output: Path) -> None:
    payload = json.loads((root / output).read_text(encoding="utf-8"))
    gates = payload.get("gates", {})
    if (
        payload.get("schema") != "trajuq_final_rq_evidence_v1"
        or payload.get("rq") != "RQ2"
        or payload.get("status") != "FINAL"
        or payload.get("experiment_commit") != EXPECTED_COMMIT
        or int(gates.get("seed_jobs_expected", -1)) != 120
        or int(gates.get("seed_jobs_complete", -1)) != 120
        or int(gates.get("partitions_expected", -1)) != 24
        or int(gates.get("partitions_complete", -1)) != 24
    ):
        raise ValueError("RQ2 candidate evidence manifest failed final gate validation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--plan", default="results/repeated_sampling_plan.json")
    parser.add_argument(
        "--state", default="results/repeated_sampling_execution_state.json"
    )
    parser.add_argument(
        "--output", default="results/generated/final/rq2_manifest.json"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    plan = Path(args.plan)
    state = Path(args.state)
    output = Path(args.output)

    try:
        if git(root, "rev-parse", "HEAD") != EXPECTED_COMMIT:
            raise ValueError("experiment checkout is not the authoritative commit")
        if git(root, "status", "--porcelain"):
            raise ValueError("experiment checkout is dirty")
        require_generation_complete(root, plan)
        run(
            [
                sys.executable,
                "scripts/execute_repeated_sampling.py",
                "--plan",
                plan.as_posix(),
                "--state-output",
                state.as_posix(),
                "--execute",
            ],
            root,
            args.dry_run,
        )
        run(
            [
                sys.executable,
                str(ROOT / "scripts/build_rq2_final_evidence.py"),
                "--root",
                str(root),
                "--plan",
                plan.as_posix(),
                "--state",
                state.as_posix(),
                "--output",
                output.as_posix(),
            ],
            root,
            args.dry_run,
        )
        if not args.dry_run:
            validate_candidate(root, output)
    except (ImportError, KeyError, OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2

    print(
        json.dumps(
            {
                "status": "PASS",
                "dry_run": args.dry_run,
                "claim_authorized": False,
                "output": output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
