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


BASE_STEPS = (
    ("generate", "generate_argv"),
    ("audit_raw", "audit_raw_argv"),
    ("score", "score_argv"),
    ("audit_scored", "audit_scored_argv"),
)


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
    stage: str,
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
        and payload.get("stage") == stage
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
    if plan.get("protocol") != "trajuq_adaptive_matrix_plan_v3":
        failures.append("adaptive plan protocol is not v3")

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
        ("primary_config", "primary_config_sha256"),
    ):
        path = Path(str(plan.get(key, "")))
        expected_hash = str(plan.get(hash_key, ""))
        if not path.exists() or not expected_hash:
            failures.append(f"adaptive plan missing bound {key} artifact")
        elif sha256_file(path) != expected_hash:
            failures.append(f"{key} hash differs from adaptive plan")

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

            refined = Path(str(spec.get("refined_scored", "")))
            refined_hash = str(spec.get("refined_scored_sha256", ""))
            audit = Path(str(spec.get("refined_audit", "")))
            audit_hash = str(spec.get("refined_audit_sha256", ""))

            if not refined.exists() or not refined_hash:
                failures.append(
                    f"{label} {partition}: bound refined scored input missing"
                )
                continue
            observed_refined_hash = sha256_file(refined)
            if observed_refined_hash != refined_hash:
                failures.append(
                    f"{label} {partition}: refined scored input hash drifted"
                )

            if not audit.exists() or not audit_hash:
                failures.append(
                    f"{label} {partition}: bound refined audit missing"
                )
                continue
            if sha256_file(audit) != audit_hash:
                failures.append(
                    f"{label} {partition}: refined audit hash drifted"
                )
                continue
            try:
                audit_payload = json.loads(audit.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                failures.append(
                    f"{label} {partition}: refined audit is invalid JSON"
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
                    f"{label} {partition}: refined audit is not a clean frozen PASS"
                )
            if str(audit_payload.get("sha256", "")) != observed_refined_hash:
                failures.append(
                    f"{label} {partition}: refined audit no longer binds input"
                )

    return failures


def base_partition_state(
    spec: dict[str, Any],
    expected_commit: str | None = None,
) -> dict[str, Any]:
    raw = Path(str(spec["raw_output"]))
    scored = Path(str(spec["scored_output"]))
    raw_ok = audit_passes(raw, "raw", expected_commit)
    scored_ok = audit_passes(scored, "scored", expected_commit)
    return {
        "raw_output": str(raw),
        "scored_output": str(scored),
        "raw_exists": raw.exists(),
        "raw_audit_pass": raw_ok,
        "scored_exists": scored.exists(),
        "scored_audit_pass": scored_ok,
        "complete": bool(raw_ok and scored_ok),
    }


def evaluation_state(spec: dict[str, Any]) -> str:
    result = Path(str(spec["result"]))
    predictions = Path(str(spec["predictions"]))
    exists = [result.exists(), predictions.exists()]
    if not any(exists):
        return "MISSING"
    if not all(exists):
        return "PARTIAL"
    if valid_json(result) and valid_jsonl(predictions):
        return "COMPLETE"
    return "INVALID"


def macro_state(spec: dict[str, Any], expected_cells: int) -> str:
    output = Path(str(spec["output"]))
    if not output.exists():
        return "MISSING"
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "INVALID"
    complete = (
        payload.get("cell_set_complete") is True
        and int(payload.get("observed_cells", -1)) == expected_cells
        and str(payload.get("fraction_key")) == str(spec.get("fraction_key"))
    )
    return "COMPLETE" if complete else "INVALID"


def cell_state(
    cell: dict[str, Any],
    fraction_keys: list[str],
    expected_commit: str | None = None,
) -> dict[str, Any]:
    calibration = base_partition_state(
        cell["calibration_base"], expected_commit
    )
    test = base_partition_state(cell["test_base"], expected_commit)
    evaluations = {
        key: evaluation_state(cell["evaluations"][key])
        for key in fraction_keys
    }
    return {
        "cell": cell["cell"],
        "adapter": cell["adapter"],
        "dataset": cell["dataset"],
        "calibration_base": calibration,
        "test_base": test,
        "evaluations": evaluations,
        "complete": bool(
            calibration["complete"]
            and test["complete"]
            and all(value == "COMPLETE" for value in evaluations.values())
        ),
    }


def execution_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    fraction_keys = [str(item["key"]) for item in plan.get("fractions", [])]
    expected = int(plan.get("expected_cells", 0))
    expected_commit = str(plan.get("experiment_git_commit", "")) or None
    cells = [
        cell_state(cell, fraction_keys, expected_commit)
        for cell in plan.get("cells", [])
    ]
    complete_cells = sum(cell["complete"] for cell in cells)

    macros = {}
    for key in fraction_keys:
        spec = dict(plan["macro_aggregation"][key])
        spec["fraction_key"] = key
        macros[key] = macro_state(spec, expected)

    status = (
        "COMPLETE"
        if len(cells) == expected
        and complete_cells == expected
        and all(value == "COMPLETE" for value in macros.values())
        else "IN_PROGRESS"
    )
    return {
        "protocol": "trajuq_adaptive_execution_state_v2",
        "status": status,
        "plan_protocol": plan.get("protocol"),
        "experiment_git_commit": plan.get("experiment_git_commit"),
        "expected_cells": expected,
        "observed_cells": len(cells),
        "complete_cells": complete_cells,
        "fraction_keys": fraction_keys,
        "macros": macros,
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


def command_needed(
    step: str,
    spec: dict[str, Any],
    expected_commit: str | None = None,
) -> bool:
    raw = Path(str(spec["raw_output"]))
    scored = Path(str(spec["scored_output"]))
    if step == "generate":
        return not raw.exists()
    if step == "audit_raw":
        return not audit_passes(raw, "raw", expected_commit)
    if step == "score":
        return not scored.exists()
    if step == "audit_scored":
        return not audit_passes(scored, "scored", expected_commit)
    raise ValueError(step)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resumably execute the frozen adaptive-compute matrix, including "
            "base-budget runs, all pre-registered caps, and macro aggregation."
        )
    )
    parser.add_argument(
        "--plan",
        default="results/adaptive_matrix_plan.json",
    )
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        help="Optional exact cell label; repeatable.",
    )
    parser.add_argument(
        "--fraction-key",
        action="append",
        default=[],
        help="Optional cap key such as cap_050; repeatable.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run missing commands. Default is dry-run/status only.",
    )
    parser.add_argument(
        "--state-output",
        default="results/adaptive_execution_state.json",
    )
    parser.add_argument(
        "--event-log",
        default="results/adaptive_execution_events.jsonl",
    )
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.exists():
        raise SystemExit(f"Missing adaptive plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "PASS":
        raise SystemExit("Adaptive plan is not PASS.")

    provenance_failures = validate_plan_provenance(plan)
    if provenance_failures:
        raise SystemExit(
            "Adaptive provenance validation failed:\n- "
            + "\n- ".join(provenance_failures)
        )

    cells = list(plan.get("cells", []))
    expected = int(plan.get("expected_cells", 0))
    if expected <= 0 or len(cells) != expected:
        raise SystemExit(
            f"Adaptive plan cell count mismatch: {len(cells)} != {expected}"
        )

    known_cells = {str(cell["cell"]) for cell in cells}
    requested_cells = set(args.cell)
    unknown_cells = sorted(requested_cells - known_cells)
    if unknown_cells:
        raise SystemExit(f"Unknown requested cells: {unknown_cells}")

    all_fraction_keys = [
        str(item["key"]) for item in plan.get("fractions", [])
    ]
    requested_fractions = set(args.fraction_key)
    unknown_fractions = sorted(
        requested_fractions - set(all_fraction_keys)
    )
    if unknown_fractions:
        raise SystemExit(
            f"Unknown requested fraction keys: {unknown_fractions}"
        )
    selected_fraction_keys = [
        key
        for key in all_fraction_keys
        if not requested_fractions or key in requested_fractions
    ]
    selected_cells = [
        cell
        for cell in cells
        if not requested_cells or str(cell["cell"]) in requested_cells
    ]

    state_path = Path(args.state_output)
    event_path = Path(args.event_log)
    expected_commit = str(plan.get("experiment_git_commit", "")) or None
    initial = write_snapshot(plan, state_path)
    print(
        "[TrajUQ] adaptive progress: "
        f"{initial['complete_cells']}/{initial['expected_cells']} cells"
    )

    for cell in selected_cells:
        scope = str(cell["cell"])
        for partition_name, spec_key in (
            ("calibration", "calibration_base"),
            ("test", "test_base"),
        ):
            spec = cell[spec_key]
            for step, argv_key in BASE_STEPS:
                if not command_needed(step, spec, expected_commit):
                    print(
                        f"[SKIP] {scope} {partition_name}_{step}: complete"
                    )
                    continue
                run_command(
                    [str(value) for value in spec[argv_key]],
                    execute=args.execute,
                    scope=scope,
                    step=f"{partition_name}_{step}",
                    event_log=event_path,
                )
                if args.execute:
                    write_snapshot(plan, state_path)

            if args.execute and not base_partition_state(
                spec, expected_commit
            )["complete"]:
                raise SystemExit(
                    f"{scope}: {partition_name} base-budget partition "
                    "did not validate."
                )

        for key in selected_fraction_keys:
            evaluation = cell["evaluations"][key]
            state = evaluation_state(evaluation)
            if state == "COMPLETE":
                print(f"[SKIP] {scope} {key}: evaluation complete")
                continue
            if state in {"PARTIAL", "INVALID"}:
                raise SystemExit(
                    f"{scope} {key}: adaptive outputs are {state}; "
                    "inspect them instead of overwriting."
                )
            run_command(
                [str(value) for value in evaluation["evaluate_argv"]],
                execute=args.execute,
                scope=scope,
                step=f"evaluate_{key}",
                event_log=event_path,
            )
            if args.execute and evaluation_state(evaluation) != "COMPLETE":
                write_snapshot(plan, state_path)
                raise SystemExit(
                    f"{scope} {key}: evaluation outputs incomplete."
                )
            if args.execute:
                write_snapshot(plan, state_path)

    after = execution_snapshot(plan)
    all_cells_complete = after["complete_cells"] == after["expected_cells"]
    subset_requested = bool(requested_cells or requested_fractions)

    for key in selected_fraction_keys:
        macro_spec = dict(plan["macro_aggregation"][key])
        macro_spec["fraction_key"] = key
        state = macro_state(macro_spec, expected)
        all_key_cells_complete = all(
            cell_state(cell, [key], expected_commit)["complete"]
            for cell in cells
        )
        if subset_requested and not all_key_cells_complete:
            print(
                f"[TrajUQ] {key} macro blocked until all 18 cells complete."
            )
            continue
        if all_key_cells_complete and state == "MISSING":
            run_command(
                [str(value) for value in macro_spec["argv"]],
                execute=args.execute,
                scope="adaptive",
                step=f"macro_{key}",
                event_log=event_path,
            )
        elif state == "COMPLETE":
            print(f"[SKIP] adaptive macro_{key}: complete")
        elif state == "INVALID":
            raise SystemExit(
                f"Existing adaptive macro {key} is invalid; "
                "do not overwrite automatically."
            )

    final = write_snapshot(plan, state_path)
    print(
        "[TrajUQ] final adaptive progress: "
        f"{final['complete_cells']}/{final['expected_cells']} cells; "
        f"macros={final['macros']}; all_cells={all_cells_complete}"
    )


if __name__ == "__main__":
    main()
