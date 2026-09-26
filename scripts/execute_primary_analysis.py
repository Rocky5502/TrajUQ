#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trajuq.io import read_jsonl


CELL_STEPS = (
    ("evaluate", "evaluate_argv", ("metrics", "predictions")),
    ("nested_comparison", "nested_comparison_argv", ("nested_bootstrap",)),
    ("feature_ablation", "feature_ablation_argv", ("feature_ablation",)),
    ("raw_risk_diagnostic", "raw_risk_diagnostic_argv", ("raw_risk_diagnostic",)),
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


def output_valid(path: Path) -> bool:
    if path.suffix == ".json":
        return valid_json(path)
    if path.suffix == ".jsonl":
        return valid_jsonl(path)
    return path.exists() and path.stat().st_size > 0


def step_state(cell: dict[str, Any], output_keys: tuple[str, ...]) -> str:
    outputs = cell["outputs"]
    paths = [Path(str(outputs[key])) for key in output_keys]
    exists = [path.exists() for path in paths]
    if not any(exists):
        return "MISSING"
    if not all(exists):
        return "PARTIAL"
    if all(output_valid(path) for path in paths):
        return "COMPLETE"
    return "INVALID"


def cell_state(cell: dict[str, Any]) -> dict[str, Any]:
    steps: dict[str, str] = {}
    for step, _, output_keys in CELL_STEPS:
        steps[step] = step_state(cell, output_keys)
    complete = all(value == "COMPLETE" for value in steps.values())
    return {
        "cell": cell["cell"],
        "adapter": cell["adapter"],
        "dataset": cell["dataset"],
        "steps": steps,
        "complete": complete,
    }


def macro_state(plan: dict[str, Any]) -> dict[str, Any]:
    macro = Path(str(plan["final_outputs"]["macro_primary"]))
    claim = Path(str(plan["final_outputs"]["rq1_claim_gate"]))

    macro_status = "MISSING"
    if macro.exists():
        if not valid_json(macro):
            macro_status = "INVALID"
        else:
            payload = json.loads(macro.read_text(encoding="utf-8"))
            macro_status = (
                "COMPLETE"
                if payload.get("cell_set_complete") is True
                else "INVALID"
            )

    claim_status = "MISSING"
    if claim.exists():
        if not valid_json(claim):
            claim_status = "INVALID"
        else:
            payload = json.loads(claim.read_text(encoding="utf-8"))
            claim_status = (
                "COMPLETE"
                if payload.get("decision")
                in {
                    "SUPPORT_POSITIVE_RQ1_CLAIM",
                    "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM",
                }
                else "INVALID"
            )

    return {
        "macro_primary": macro_status,
        "rq1_claim_gate": claim_status,
    }


def execution_snapshot(plan: dict[str, Any]) -> dict[str, Any]:
    cells = [cell_state(cell) for cell in plan.get("cell_analyses", [])]
    complete_cells = sum(cell["complete"] for cell in cells)
    final = macro_state(plan)
    expected = int(plan.get("cell_count", len(cells)))
    status = (
        "COMPLETE"
        if len(cells) == expected
        and complete_cells == expected
        and final["macro_primary"] == "COMPLETE"
        and final["rq1_claim_gate"] == "COMPLETE"
        else "IN_PROGRESS"
    )
    return {
        "protocol": "trajuq_primary_analysis_execution_state_v1",
        "status": status,
        "expected_cells": expected,
        "observed_cells": len(cells),
        "complete_cells": complete_cells,
        "cells": cells,
        "final": final,
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
    print(
        f"[{'RUN' if execute else 'DRY'}] {scope} {step}: "
        + " ".join(argv)
    )
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
            "Resumably execute the frozen primary-analysis ledger without "
            "overwriting partial or existing outputs."
        )
    )
    parser.add_argument(
        "--plan",
        default="results/primary_analysis_plan.json",
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
        default="results/primary_analysis_execution_state.json",
    )
    parser.add_argument(
        "--event-log",
        default="results/primary_analysis_execution_events.jsonl",
    )
    args = parser.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.exists():
        raise SystemExit(f"Missing analysis plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "PASS":
        raise SystemExit("Primary analysis plan is not PASS.")

    cells = list(plan.get("cell_analyses", []))
    expected = int(plan.get("cell_count", 0))
    if expected <= 0 or len(cells) != expected:
        raise SystemExit(
            f"Analysis plan cell count mismatch: {len(cells)} != {expected}"
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
    initial = write_snapshot(plan, state_path)
    print(
        "[TrajUQ] analysis progress: "
        f"{initial['complete_cells']}/{initial['expected_cells']} cells"
    )

    for cell in selected:
        scope = str(cell["cell"])
        for step, argv_key, output_keys in CELL_STEPS:
            state = step_state(cell, output_keys)
            if state == "COMPLETE":
                print(f"[SKIP] {scope} {step}: outputs already valid")
                continue
            if state in {"PARTIAL", "INVALID"}:
                raise SystemExit(
                    f"{scope} {step}: existing outputs are {state}; "
                    "inspect them manually rather than overwriting."
                )

            argv = [str(value) for value in cell[argv_key]]
            run_command(
                argv,
                execute=args.execute,
                scope=scope,
                step=step,
                event_log=event_path,
            )
            if args.execute:
                after = step_state(cell, output_keys)
                if after != "COMPLETE":
                    write_snapshot(plan, state_path)
                    raise SystemExit(
                        f"{scope} {step}: command returned but outputs are {after}"
                    )
                write_snapshot(plan, state_path)

    full_after_cells = execution_snapshot(plan)
    all_cells_complete = (
        full_after_cells["complete_cells"]
        == full_after_cells["expected_cells"]
    )

    if requested and not all_cells_complete:
        print(
            "[TrajUQ] subset execution complete; final macro/claim gate "
            "remain blocked until every cell is complete."
        )
        write_snapshot(plan, state_path)
        return

    if all_cells_complete:
        final = macro_state(plan)
        if final["macro_primary"] == "MISSING":
            run_command(
                [str(value) for value in plan["macro_aggregation_argv"]],
                execute=args.execute,
                scope="primary",
                step="macro_aggregation",
                event_log=event_path,
            )
            if args.execute and macro_state(plan)["macro_primary"] != "COMPLETE":
                write_snapshot(plan, state_path)
                raise SystemExit("Primary macro command did not produce a complete macro.")
        elif final["macro_primary"] == "INVALID":
            raise SystemExit(
                "Existing primary macro is invalid; do not overwrite it automatically."
            )
        else:
            print("[SKIP] primary macro_aggregation: complete")

        final = macro_state(plan)
        if final["macro_primary"] == "COMPLETE":
            if final["rq1_claim_gate"] == "MISSING":
                run_command(
                    [str(value) for value in plan["rq1_claim_gate_argv"]],
                    execute=args.execute,
                    scope="primary",
                    step="rq1_claim_gate",
                    event_log=event_path,
                )
            elif final["rq1_claim_gate"] == "INVALID":
                raise SystemExit(
                    "Existing RQ1 claim-gate artifact is invalid; "
                    "do not overwrite it automatically."
                )
            else:
                print("[SKIP] primary rq1_claim_gate: complete")

    final_snapshot = write_snapshot(plan, state_path)
    print(
        "[TrajUQ] final analysis progress: "
        f"{final_snapshot['complete_cells']}/{final_snapshot['expected_cells']} cells; "
        f"macro={final_snapshot['final']['macro_primary']}; "
        f"claim_gate={final_snapshot['final']['rq1_claim_gate']}"
    )


if __name__ == "__main__":
    main()
