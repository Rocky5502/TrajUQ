#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trajuq.io import read_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def git_status_porcelain() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def audit_passes(
    path: Path,
    expected_commit: str | None = None,
) -> bool:
    audit = path.with_suffix(path.suffix + ".audit.json")
    if not path.exists() or not audit.exists():
        return False
    try:
        payload = json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("protocol") == "trajuq_run_audit_v1"
        and payload.get("status") == "PASS"
        and payload.get("git_clean") is True
        and payload.get("stage") == "raw"
        and str(payload.get("sha256", "")) == sha256_file(path)
        and (
            expected_commit is None
            or payload.get("git_commit") == expected_commit
        )
    )


def valid_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def valid_jsonl(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rows = list(read_jsonl(path))
    except Exception:
        return False
    return bool(rows)


def validate_plan_provenance(
    plan: dict[str, Any],
    *,
    require_current_commit: bool = True,
) -> list[str]:
    failures: list[str] = []
    if plan.get("protocol") != "trajuq_repeated_sampling_plan_v2":
        failures.append("repeated-sampling plan protocol is not v2")

    expected_commit = str(plan.get("experiment_git_commit", ""))
    if require_current_commit:
        current_head = git_head()
        if not current_head:
            failures.append("current git commit could not be resolved")
        elif current_head != expected_commit:
            failures.append(
                "current git commit differs from the frozen experiment commit"
            )
        current_status = git_status_porcelain()
        if current_status is None:
            failures.append("current git worktree status could not be resolved")
        elif current_status:
            failures.append(
                "current git worktree is dirty; frozen execution requires a clean checkout"
            )

    for key, hash_key in (
        ("primary_plan", "primary_plan_sha256"),
        ("primary_analysis_plan", "primary_analysis_plan_sha256"),
        ("primary_config", "primary_config_sha256"),
    ):
        path = Path(str(plan.get(key, "")))
        expected_hash = str(plan.get(hash_key, ""))
        if not path.exists() or not expected_hash:
            failures.append(f"repeated plan missing bound {key} artifact")
        elif sha256_file(path) != expected_hash:
            failures.append(f"{key} hash differs from repeated plan")

    for cell in plan.get("cells", []):
        label = str(cell.get("cell", "<unknown>"))
        provenance = cell.get("provenance", {})
        for partition in ("calibration", "test"):
            spec = provenance.get(partition, {})
            generation_input = Path(str(spec.get("generation_input", "")))
            input_hash = str(spec.get("generation_input_sha256", ""))
            if not generation_input.exists() or not input_hash:
                failures.append(
                    f"{label} {partition}: bound generation input missing"
                )
            elif sha256_file(generation_input) != input_hash:
                failures.append(
                    f"{label} {partition}: generation input hash drifted"
                )

            scored = Path(str(spec.get("scored_target", "")))
            scored_hash = str(spec.get("scored_target_sha256", ""))
            audit = Path(str(spec.get("scored_audit", "")))
            audit_hash = str(spec.get("scored_audit_sha256", ""))
            if not scored.exists() or not scored_hash:
                failures.append(
                    f"{label} {partition}: bound scored target missing"
                )
                continue
            observed_scored_hash = sha256_file(scored)
            if observed_scored_hash != scored_hash:
                failures.append(
                    f"{label} {partition}: scored target hash drifted"
                )

            if not audit.exists() or not audit_hash:
                failures.append(f"{label} {partition}: bound scored audit missing")
                continue
            if sha256_file(audit) != audit_hash:
                failures.append(
                    f"{label} {partition}: scored audit hash drifted"
                )
                continue
            try:
                audit_payload = json.loads(audit.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                failures.append(
                    f"{label} {partition}: scored audit is invalid JSON"
                )
                continue
            if (
                audit_payload.get("protocol") != "trajuq_run_audit_v1"
                or audit_payload.get("status") != "PASS"
                or audit_payload.get("git_clean") is not True
                or audit_payload.get("git_commit") != expected_commit
                or audit_payload.get("stage") != "scored"
            ):
                failures.append(
                    f"{label} {partition}: scored audit is not a clean frozen PASS"
                )
            if str(audit_payload.get("sha256", "")) != observed_scored_hash:
                failures.append(
                    f"{label} {partition}: scored audit no longer binds target"
                )

        trajuq_predictions = Path(
            str(provenance.get("trajuq_predictions", ""))
        )
        predictions_hash = str(
            provenance.get("trajuq_predictions_sha256", "")
        )
        if not valid_jsonl(trajuq_predictions) or not predictions_hash:
            failures.append(
                f"{label}: bound TrajUQ predictions are missing or invalid"
            )
        elif sha256_file(trajuq_predictions) != predictions_hash:
            failures.append(f"{label}: TrajUQ predictions hash drifted")

    return failures


def generation_dir_from_argv(argv: list[str]) -> Path:
    if "--output-dir" not in argv:
        raise ValueError("Repeated generation argv has no --output-dir.")
    index = argv.index("--output-dir")
    return Path(str(argv[index + 1]))


def generation_state(
    argv: list[str],
    seeds: list[int],
    expected_commit: str | None = None,
) -> dict[str, Any]:
    root = generation_dir_from_argv(argv)
    seed_state = {}
    for seed in seeds:
        path = root / f"seed-{seed}.raw.jsonl"
        seed_state[str(seed)] = {
            "path": str(path),
            "complete": audit_passes(path, expected_commit),
        }
    manifest = root / "manifest.json"
    complete = (
        all(item["complete"] for item in seed_state.values())
        and valid_json(manifest)
    )
    return {
        "output_dir": str(root),
        "manifest": str(manifest),
        "seeds": seed_state,
        "complete": complete,
    }


def pair_state(first: Path, second: Path) -> str:
    exists = [first.exists(), second.exists()]
    if not any(exists):
        return "MISSING"
    if not all(exists):
        return "PARTIAL"
    if valid_json(first) and valid_jsonl(second):
        return "COMPLETE"
    return "INVALID"


def comparison_state(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return "COMPLETE" if valid_json(path) else "INVALID"


def cell_state(
    cell: dict[str, Any],
    expected_commit: str | None = None,
) -> dict[str, Any]:
    seeds = [int(seed) for seed in cell["seeds"]]
    calibration = generation_state(
        [str(value) for value in cell["calibration_generation_argv"]],
        seeds,
        expected_commit,
    )
    test = generation_state(
        [str(value) for value in cell["test_generation_argv"]],
        seeds,
        expected_commit,
    )
    evaluation = pair_state(
        Path(str(cell["repeated_metrics"])),
        Path(str(cell["repeated_predictions"])),
    )
    comparison = comparison_state(Path(str(cell["direct_comparison"])))
    return {
        "cell": cell["cell"],
        "adapter": cell["adapter"],
        "dataset": cell["dataset"],
        "calibration_generation": calibration,
        "test_generation": test,
        "evaluation": evaluation,
        "comparison": comparison,
        "complete": bool(
            calibration["complete"]
            and test["complete"]
            and evaluation == "COMPLETE"
            and comparison == "COMPLETE"
        ),
    }


def macro_state(plan: dict[str, Any]) -> str:
    path = Path(str(plan["macro_output"]))
    if not path.exists():
        return "MISSING"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "INVALID"
    complete = (
        payload.get("cell_set_complete") is True
        and int(payload.get("observed_cells", -1))
        == int(plan.get("expected_cells", 0))
    )
    return "COMPLETE" if complete else "INVALID"


def execution_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    expected_commit = str(plan.get("experiment_git_commit", "")) or None
    cells = [
        cell_state(cell, expected_commit)
        for cell in plan.get("cells", [])
    ]
    expected = int(plan.get("expected_cells", 0))
    complete_cells = sum(cell["complete"] for cell in cells)
    macro = macro_state(plan)
    status = (
        "COMPLETE"
        if len(cells) == expected
        and complete_cells == expected
        and macro == "COMPLETE"
        else "IN_PROGRESS"
    )
    return {
        "protocol": "trajuq_repeated_execution_state_v2",
        "status": status,
        "plan_protocol": plan.get("protocol"),
        "experiment_git_commit": plan.get("experiment_git_commit"),
        "expected_cells": expected,
        "observed_cells": len(cells),
        "complete_cells": complete_cells,
        "macro": macro,
        "cells": cells,
    }


def write_snapshot(plan: dict[str, Any], output: Path) -> dict[str, Any]:
    snapshot = execution_snapshot(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot


def append_event(
    path: Path,
    *,
    scope: str,
    step: str,
    argv: list[str],
    event: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "step": step,
        "event": event,
        "argv": argv,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_command(
    argv: list[str],
    *,
    execute: bool,
    scope: str,
    step: str,
    event_log: Path,
) -> None:
    mode = "RUN" if execute else "DRY"
    print(f"[{mode}] {scope} {step}: " + " ".join(argv))
    if not execute:
        return
    append_event(
        event_log,
        scope=scope,
        step=step,
        argv=argv,
        event="START",
    )
    try:
        subprocess.run(argv, check=True)
    except subprocess.CalledProcessError:
        append_event(
            event_log,
            scope=scope,
            step=step,
            argv=argv,
            event="FAIL",
        )
        raise
    append_event(
        event_log,
        scope=scope,
        step=step,
        argv=argv,
        event="PASS",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resumably execute the frozen five-sample baseline. Existing seed "
            "runs are reused only when raw audit and SHA-256 checks pass."
        )
    )
    parser.add_argument(
        "--plan",
        default="results/repeated_sampling_plan.json",
    )
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        help="Optional exact cell label; repeatable.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run missing commands. Default is dry-run/status only.",
    )
    parser.add_argument(
        "--state-output",
        default="results/repeated_sampling_execution_state.json",
    )
    parser.add_argument(
        "--event-log",
        default="results/repeated_sampling_execution_events.jsonl",
    )
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.exists():
        raise SystemExit(f"Missing repeated-sampling plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "PASS":
        raise SystemExit("Repeated-sampling plan is not PASS.")

    provenance_failures = validate_plan_provenance(plan)
    if provenance_failures:
        raise SystemExit(
            "Repeated-sampling provenance validation failed:\n- "
            + "\n- ".join(provenance_failures)
        )

    cells = list(plan.get("cells", []))
    expected = int(plan.get("expected_cells", 0))
    if expected <= 0 or len(cells) != expected:
        raise SystemExit(
            f"Repeated plan cell count mismatch: {len(cells)} != {expected}"
        )

    requested = set(args.cell)
    known = {str(cell["cell"]) for cell in cells}
    unknown = sorted(requested - known)
    if unknown:
        raise SystemExit(f"Unknown requested cells: {unknown}")
    selected = [
        cell for cell in cells
        if not requested or str(cell["cell"]) in requested
    ]

    state_path = Path(args.state_output)
    event_path = Path(args.event_log)
    expected_commit = str(plan.get("experiment_git_commit", "")) or None
    initial = write_snapshot(plan, state_path)
    print(
        "[TrajUQ] repeated progress: "
        f"{initial['complete_cells']}/{initial['expected_cells']} cells"
    )

    for cell in selected:
        scope = str(cell["cell"])
        seeds = [int(seed) for seed in cell["seeds"]]

        for partition, argv_key in (
            ("calibration", "calibration_generation_argv"),
            ("test", "test_generation_argv"),
        ):
            argv = [str(value) for value in cell[argv_key]]
            state = generation_state(argv, seeds, expected_commit)
            if state["complete"]:
                print(f"[SKIP] {scope} {partition}_generation: complete")
                continue
            run_command(
                argv,
                execute=args.execute,
                scope=scope,
                step=f"{partition}_generation",
                event_log=event_path,
            )
            if args.execute and not generation_state(
                argv, seeds, expected_commit
            )["complete"]:
                write_snapshot(plan, state_path)
                raise SystemExit(
                    f"{scope}: {partition} repeated generation incomplete."
                )
            if args.execute:
                write_snapshot(plan, state_path)

        evaluation = pair_state(
            Path(str(cell["repeated_metrics"])),
            Path(str(cell["repeated_predictions"])),
        )
        if evaluation == "MISSING":
            run_command(
                [str(value) for value in cell["evaluate_argv"]],
                execute=args.execute,
                scope=scope,
                step="evaluate",
                event_log=event_path,
            )
            if args.execute and pair_state(
                Path(str(cell["repeated_metrics"])),
                Path(str(cell["repeated_predictions"])),
            ) != "COMPLETE":
                write_snapshot(plan, state_path)
                raise SystemExit(
                    f"{scope}: repeated evaluation outputs incomplete."
                )
            if args.execute:
                write_snapshot(plan, state_path)
        elif evaluation == "COMPLETE":
            print(f"[SKIP] {scope} evaluate: complete")
        else:
            raise SystemExit(
                f"{scope}: repeated evaluation outputs are {evaluation}; "
                "inspect them instead of overwriting."
            )

        comparison = comparison_state(Path(str(cell["direct_comparison"])))
        if comparison == "MISSING":
            run_command(
                [str(value) for value in cell["compare_to_trajuq_argv"]],
                execute=args.execute,
                scope=scope,
                step="compare_to_trajuq",
                event_log=event_path,
            )
            if args.execute and comparison_state(
                Path(str(cell["direct_comparison"]))
            ) != "COMPLETE":
                write_snapshot(plan, state_path)
                raise SystemExit(
                    f"{scope}: direct comparison output is incomplete."
                )
            if args.execute:
                write_snapshot(plan, state_path)
        elif comparison == "COMPLETE":
            print(f"[SKIP] {scope} compare_to_trajuq: complete")
        else:
            raise SystemExit(
                f"{scope}: comparison output is {comparison}; "
                "inspect it instead of overwriting."
            )

    after = execution_snapshot(plan)
    all_cells_complete = after["complete_cells"] == after["expected_cells"]
    if requested and not all_cells_complete:
        write_snapshot(plan, state_path)
        print(
            "[TrajUQ] subset complete; macro remains blocked until all cells "
            "finish."
        )
        return

    macro = macro_state(plan)
    if all_cells_complete and macro == "MISSING":
        run_command(
            [str(value) for value in plan["macro_aggregation_argv"]],
            execute=args.execute,
            scope="repeated",
            step="macro_aggregation",
            event_log=event_path,
        )
    elif macro == "COMPLETE":
        print("[SKIP] repeated macro_aggregation: complete")
    elif macro == "INVALID":
        raise SystemExit(
            "Existing repeated macro is invalid; "
            "do not overwrite automatically."
        )

    final = write_snapshot(plan, state_path)
    print(
        "[TrajUQ] final repeated progress: "
        f"{final['complete_cells']}/{final['expected_cells']} cells; "
        f"macro={final['macro']}"
    )


if __name__ == "__main__":
    main()
