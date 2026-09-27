#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.execute_adaptive_plan import (  # noqa: E402
    artifact_id_state as adaptive_artifact_id_state,
    validate_plan_provenance as validate_adaptive_plan_provenance,
)
from scripts.execute_primary_plan import (  # noqa: E402
    artifact_id_state as primary_artifact_id_state,
    validate_plan_provenance as validate_primary_plan_provenance,
)
from scripts.run_generation import build_adapter, generate_record  # noqa: E402
from trajuq.io import append_jsonl, existing_ids, read_jsonl  # noqa: E402


def argv_value(
    argv: list[str],
    flag: str,
    default: Any = None,
    cast: Callable[[str], Any] = str,
) -> Any:
    if flag not in argv:
        return default
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise ValueError(f"frozen generation argv has no value for {flag}")
    return cast(str(argv[index + 1]))


def generation_namespace(spec: dict[str, Any]) -> SimpleNamespace:
    argv = [str(value) for value in spec["generate_argv"]]
    if len(argv) < 3 or argv[1] != "scripts/run_generation.py":
        raise ValueError("generation argv is not the frozen run_generation command")

    adapter = argv_value(argv, "--adapter")
    model = argv_value(argv, "--model")
    revision = argv_value(argv, "--revision")
    input_path = argv_value(argv, "--input")
    output = argv_value(argv, "--output")
    seed = argv_value(argv, "--seed", 2027, int)
    if not all(str(value).strip() for value in (adapter, model, revision, input_path, output)):
        raise ValueError("generation argv is missing frozen model/input/output identity")

    return SimpleNamespace(
        adapter=str(adapter),
        model=str(model),
        revision=str(revision),
        input=str(input_path),
        output=str(output),
        seed=int(seed),
        device=str(argv_value(argv, "--device", "cuda")),
        dtype=str(argv_value(argv, "--dtype", "bfloat16")),
        mask_id=argv_value(argv, "--mask-id", None, int),
        steps=int(argv_value(argv, "--steps", 128, int)),
        max_new_tokens=int(argv_value(argv, "--max-new-tokens", 128, int)),
        block_length=int(argv_value(argv, "--block-length", 128, int)),
        temperature=float(argv_value(argv, "--temperature", 0.0, float)),
        top_p=float(argv_value(argv, "--top-p", 1.0, float)),
        llada_cfg_scale=float(argv_value(argv, "--llada-cfg-scale", 0.0, float)),
        llada_remasking=str(
            argv_value(argv, "--llada-remasking", "low_confidence")
        ),
        dream_alg=str(argv_value(argv, "--dream-alg", "entropy")),
        dream_alg_temp=float(argv_value(argv, "--dream-alg-temp", 0.1, float)),
        sdar_denoising_steps=int(
            argv_value(argv, "--sdar-denoising-steps", 4, int)
        ),
        sdar_temperature=argv_value(argv, "--sdar-temperature", None, float),
        sdar_top_k=int(argv_value(argv, "--sdar-top-k", 0, int)),
        sdar_top_p=argv_value(argv, "--sdar-top-p", None, float),
        sdar_remasking_strategy=str(
            argv_value(
                argv,
                "--sdar-remasking-strategy",
                "low_confidence_dynamic",
            )
        ),
        sdar_confidence_threshold=float(
            argv_value(argv, "--sdar-confidence-threshold", 0.85, float)
        ),
    )


def collect_generation_jobs(
    plan: dict[str, Any],
    kind: str,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for cell in plan.get("cells", []):
        label = str(cell["cell"])
        adapter = str(cell["adapter"])
        if kind == "primary":
            partitions = (
                ("calibration", cell["calibration"]),
                ("test", cell["test"]),
            )
        elif kind == "adaptive":
            partitions = (
                ("calibration", cell["calibration_base"]),
                ("test", cell["test_base"]),
            )
        else:
            raise ValueError(f"unknown resident-generation plan kind: {kind}")

        for partition, spec in partitions:
            args = generation_namespace(spec)
            if args.adapter != adapter:
                raise ValueError(
                    f"{label} {partition}: adapter differs from cell identity"
                )
            jobs.append(
                {
                    "cell": label,
                    "adapter": adapter,
                    "partition": partition,
                    "spec": spec,
                    "args": args,
                }
            )
    return jobs


def validate_plan(plan: dict[str, Any], kind: str) -> None:
    if plan.get("status") != "PASS":
        raise SystemExit(f"{kind} plan is not PASS")
    if kind == "primary":
        failures = validate_primary_plan_provenance(plan)
    elif kind == "adaptive":
        failures = validate_adaptive_plan_provenance(plan)
    else:
        raise SystemExit(f"Unknown plan kind: {kind}")
    if failures:
        raise SystemExit(
            f"{kind} plan provenance validation failed:\n- "
            + "\n- ".join(failures)
        )


def state_for_job(
    job: dict[str, Any],
    kind: str,
    expected_commit: str,
) -> str:
    path = Path(str(job["spec"]["raw_output"]))
    if kind == "primary":
        return primary_artifact_id_state(path, job["spec"], expected_commit)
    return adaptive_artifact_id_state(path, job["spec"], expected_commit)


def generate_missing_jobs(
    plan: dict[str, Any],
    *,
    kind: str,
    only_adapter: set[str] | None = None,
) -> None:
    validate_plan(plan, kind)
    expected_commit = str(plan.get("experiment_git_commit", "")).strip()
    if not expected_commit:
        raise SystemExit("frozen plan has no experiment_git_commit")

    jobs = collect_generation_jobs(plan, kind)
    if only_adapter:
        jobs = [job for job in jobs if job["adapter"] in only_adapter]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        state = state_for_job(job, kind, expected_commit)
        if state == "INVALID":
            raise SystemExit(
                f"{job['cell']} {job['partition']}: raw artifact is INVALID; "
                "inspect it instead of attempting resident resume"
            )
        if state != "COMPLETE":
            grouped[job["adapter"]].append(job)
        else:
            print(
                f"[RESIDENT SKIP] {job['cell']} {job['partition']}: raw complete"
            )

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
                f"{adapter_name}: resident jobs disagree on model load identity"
            )

        print(
            f"[RESIDENT LOAD] adapter={adapter_name} model={first.model} "
            f"jobs={len(adapter_jobs)}"
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
            spec = job["spec"]
            args = job["args"]
            raw_path = Path(str(spec["raw_output"]))
            state = state_for_job(job, kind, expected_commit)
            if state == "INVALID":
                raise SystemExit(
                    f"{job['cell']} {job['partition']}: raw artifact became INVALID"
                )
            if state == "COMPLETE":
                continue

            rows = list(read_jsonl(spec["input"]))
            done = existing_ids(raw_path)
            expected_prefix = [str(row["id"]) for row in rows[: len(done)]]
            if list(done) and set(done) != set(expected_prefix):
                # state_for_job already validates ordered prefix. This additional
                # guard prevents accidental reliance on unordered set semantics.
                observed = [
                    str(row.get("example_id", ""))
                    for row in read_jsonl(raw_path)
                ]
                if observed != expected_prefix:
                    raise SystemExit(
                        f"{job['cell']} {job['partition']}: existing raw rows "
                        "are not the frozen input prefix"
                    )

            for index, row in enumerate(rows):
                example_id = str(row["id"])
                if example_id in done:
                    continue
                record = generate_record(
                    resident_adapter,
                    args,
                    row,
                    seed=args.seed,
                    source_git_commit=expected_commit,
                    source_git_clean=True,
                    torch=torch,
                )
                append_jsonl(raw_path, record.to_dict())
                done.add(example_id)
                print(
                    f"[RESIDENT] {job['cell']} {job['partition']} "
                    f"{index + 1}/{len(rows)} wrote {example_id}"
                )

            final_state = state_for_job(job, kind, expected_commit)
            if final_state != "COMPLETE":
                raise SystemExit(
                    f"{job['cell']} {job['partition']}: resident generation "
                    f"finished with state={final_state}"
                )

        del resident_adapter
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        f"[TrajUQ] resident raw generation complete for kind={kind}; "
        "run the canonical executor next for audits/scoring/state validation."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Accelerate frozen primary/adaptive raw generation by keeping one "
            "model resident across all missing partitions for that adapter. "
            "Canonical executors remain responsible for audits and scoring."
        )
    )
    parser.add_argument("--kind", choices=["primary", "adaptive"], required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        choices=["llada", "dream", "sdar"],
        help="Optional adapter filter; repeatable.",
    )
    args = parser.parse_args()

    path = Path(args.plan)
    if not path.is_file():
        raise SystemExit(f"Missing frozen plan: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    generate_missing_jobs(
        plan,
        kind=args.kind,
        only_adapter=set(args.adapter) or None,
    )


if __name__ == "__main__":
    main()
