#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.execute_repeated_sampling import (  # noqa: E402
    generation_state,
    validate_plan_provenance,
)
from scripts.run_generation import build_adapter  # noqa: E402
from scripts.run_repeated_samples import (  # noqa: E402
    build_parser as build_repeated_parser,
    run_repeated_generation,
)


def parse_repeated_argv(argv: list[str]):
    command = [str(value) for value in argv]
    if len(command) < 3 or command[1] != "scripts/run_repeated_samples.py":
        raise ValueError(
            "frozen repeated generation argv is not run_repeated_samples.py"
        )
    args = build_repeated_parser().parse_args(command[2:])
    if not args.resume_verified or not args.resume_partial:
        raise ValueError(
            "frozen repeated generation argv must enable verified and partial resume"
        )
    return args


def collect_jobs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for cell in plan.get("cells", []):
        label = str(cell.get("cell", ""))
        adapter = str(cell.get("adapter", ""))
        seeds = [int(seed) for seed in cell.get("seeds", [])]
        if not label or not adapter or not seeds:
            raise ValueError("repeated plan cell lacks identity or seeds")
        for partition, key in (
            ("calibration", "calibration_generation_argv"),
            ("test", "test_generation_argv"),
        ):
            argv = [str(value) for value in cell.get(key, [])]
            args = parse_repeated_argv(argv)
            if args.adapter != adapter:
                raise ValueError(
                    f"{label} {partition}: adapter differs from cell identity"
                )
            if [int(seed) for seed in args.seeds] != seeds:
                raise ValueError(
                    f"{label} {partition}: seed set differs from frozen cell"
                )
            jobs.append(
                {
                    "cell": label,
                    "partition": partition,
                    "adapter": adapter,
                    "argv": argv,
                    "args": args,
                    "seeds": seeds,
                }
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute missing repeated-sampling generation jobs while keeping "
            "one checkpoint resident across all frozen eligible partitions. "
            "The canonical repeated executor still validates/evaluates/aggregates."
        )
    )
    parser.add_argument(
        "--plan",
        default="results/repeated_sampling_plan.json",
    )
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.is_file():
        raise SystemExit(f"Missing repeated-sampling plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "PASS":
        raise SystemExit("Repeated-sampling plan is not PASS.")

    failures = validate_plan_provenance(plan)
    if failures:
        raise SystemExit(
            "Repeated-sampling plan provenance check failed:\n- "
            + "\n- ".join(failures)
        )

    expected_commit = str(plan.get("experiment_git_commit", "")) or None
    jobs = collect_jobs(plan)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        state = generation_state(
            job["argv"],
            job["seeds"],
            expected_commit,
        )
        if state["complete"]:
            print(
                f"[REPEATED RESIDENT SKIP] {job['cell']} "
                f"{job['partition']}: generation complete"
            )
            continue
        grouped[job["adapter"]].append(job)

    torch = None
    try:
        import torch as _torch

        torch = _torch
    except ImportError:
        pass

    for adapter_name in sorted(grouped):
        adapter_jobs = grouped[adapter_name]
        first = adapter_jobs[0]["args"]
        identities = {
            (
                job["args"].model,
                job["args"].revision,
                job["args"].device,
                job["args"].dtype,
                job["args"].mask_id,
            )
            for job in adapter_jobs
        }
        if len(identities) != 1:
            raise SystemExit(
                f"{adapter_name}: repeated jobs disagree on model load identity"
            )

        print(
            f"[REPEATED RESIDENT LOAD] adapter={adapter_name} "
            f"model={first.model} jobs={len(adapter_jobs)}"
        )
        resident_adapter = build_adapter(
            first.adapter,
            first.model,
            first.device,
            first.dtype,
            first.mask_id,
            first.revision,
        )

        for job in adapter_jobs:
            print(
                f"[REPEATED RESIDENT JOB] {job['cell']} "
                f"{job['partition']}"
            )
            run_repeated_generation(
                job["args"],
                resident_adapter=resident_adapter,
                torch_module=torch,
            )
            final_state = generation_state(
                job["argv"],
                job["seeds"],
                expected_commit,
            )
            if not final_state["complete"]:
                raise SystemExit(
                    f"{job['cell']} {job['partition']}: generation "
                    "did not validate after resident execution"
                )

        del resident_adapter
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        "[TrajUQ] resident repeated generation complete; run the canonical "
        "repeated executor for evaluation, comparisons, aggregation, and state."
    )


if __name__ == "__main__":
    main()
