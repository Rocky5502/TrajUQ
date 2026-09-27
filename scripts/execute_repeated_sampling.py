#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from trajuq.io import read_jsonl
from trajuq.provenance import binding_matches


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


def argv_value(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    idx = argv.index(flag)
    if idx + 1 >= len(argv):
        return None
    return str(argv[idx + 1])


def expected_ids_from_generation_argv(argv: list[str]) -> list[str]:
    input_text = argv_value(argv, "--input")
    if not input_text:
        raise ValueError("repeated generation argv has no --input")
    rows = list(read_jsonl(Path(input_text)))
    limit_text = argv_value(argv, "--limit")
    if limit_text is not None:
        rows = rows[: int(limit_text)]
    ids = [str(row.get("id", "")) for row in rows]
    if not rows or any(not item_id for item_id in ids):
        raise ValueError(f"invalid repeated generation input IDs: {input_text}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate repeated generation input IDs: {input_text}")
    return ids


def artifact_example_ids(path: Path) -> list[str]:
    rows = list(read_jsonl(path))
    ids = [str(row.get("example_id", "")) for row in rows]
    if not rows or any(not item_id for item_id in ids):
        raise ValueError(f"invalid example IDs: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate example IDs: {path}")
    return ids


def exact_example_id_coverage(path: Path, expected_ids: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        return artifact_example_ids(path) == expected_ids
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return False


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
    try:
        expected_ids = expected_ids_from_generation_argv(argv)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        expected_ids = []

    seed_state = {}
    for seed in seeds:
        path = root / f"seed-{seed}.raw.jsonl"
        ids_complete = bool(expected_ids) and exact_example_id_coverage(
            path,
            expected_ids,
        )
        seed_state[str(seed)] = {
            "path": str(path),
            "ids_complete": ids_complete,
            "audit_pass": audit_passes(path, expected_commit),
            "complete": bool(
                ids_complete and audit_passes(path, expected_commit)
            ),
        }

    manifest = root / "manifest.json"
    manifest_ok = False
    if valid_json(manifest):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        input_text = argv_value(argv, "--input")
        input_path = Path(input_text) if input_text else None
        run_rows = list(payload.get("runs", []))
        run_seeds = [int(item.get("seed")) for item in run_rows if item.get("seed") is not None]
        manifest_ok = (
            payload.get("protocol") == "repeated_sampling_v1"
            and payload.get("git_clean") is True
            and (
                expected_commit is None
                or payload.get("git_commit") == expected_commit
            )
            and input_path is not None
            and input_path.exists()
            and str(payload.get("input")) == str(input_path)
            and str(payload.get("input_sha256", "")) == sha256_file(input_path)
            and [int(seed) for seed in payload.get("seeds", [])] == seeds
            and run_seeds == seeds
        )

    complete = (
        bool(expected_ids)
        and all(item["complete"] for item in seed_state.values())
        and manifest_ok
    )
    return {
        "output_dir": str(root),
        "manifest": str(manifest),
        "manifest_valid": manifest_ok,
        "expected_records": len(expected_ids),
        "seeds": seed_state,
        "complete": complete,
    }


def pair_state(
    first: Path,
    second: Path,
    expected_ids: list[str] | None = None,
) -> str:
    exists = [first.exists(), second.exists()]
    if not any(exists):
        return "MISSING"
    if not all(exists):
        return "PARTIAL"
    if not valid_json(first) or not valid_jsonl(second):
        return "INVALID"
    if expected_ids is not None and not exact_example_id_coverage(
        second,
        expected_ids,
    ):
        return "INVALID"
    try:
        metrics = json.loads(first.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "INVALID"
    if metrics.get("protocol") != "trajuq_repeated_sampling_baseline_v2":
        return "INVALID"
    if int(metrics.get("sample_count", -1)) != 5:
        return "INVALID"
    if expected_ids is not None and int(metrics.get("n_test", -1)) != len(expected_ids):
        return "INVALID"

    test_cost = metrics.get("auxiliary_compute", {}).get("test", {})
    for key in (
        "mean_auxiliary_wall_time_seconds_per_example",
        "p90_auxiliary_wall_time_seconds_per_example",
    ):
        try:
            value = float(test_cost.get(key))
        except (TypeError, ValueError):
            return "INVALID"
        if not math.isfinite(value) or value < 0:
            return "INVALID"
    return "COMPLETE"


def comparison_state(
    path: Path,
    expected_n: int | None = None,
) -> str:
    if not path.exists():
        return "MISSING"
    if not valid_json(path):
        return "INVALID"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "INVALID"
    if payload.get("protocol") != "trajuq_vs_repeated_sampling_v1":
        return "INVALID"
    if expected_n is not None and int(payload.get("n", -1)) != expected_n:
        return "INVALID"
    if set(payload.get("metrics", {})) < {"auroc", "auprc", "aurc"}:
        return "INVALID"
    return "COMPLETE"


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
    try:
        expected_test_ids = artifact_example_ids(
            Path(str(cell["provenance"]["test"]["scored_target"]))
        )
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        expected_test_ids = []
    evaluation = pair_state(
        Path(str(cell["repeated_metrics"])),
        Path(str(cell["repeated_predictions"])),
        expected_test_ids,
    )
    comparison = comparison_state(
        Path(str(cell["direct_comparison"])),
        len(expected_test_ids),
    )
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


def repeated_macro_provenance_matches(
    plan: dict[str, Any],
    payload: dict,
) -> bool:
    # Lightweight unit fixtures may omit the frozen-plan protocol.
    if plan.get("protocol") != "trajuq_repeated_sampling_plan_v2":
        return True

    expected_commit = str(plan.get("experiment_git_commit", ""))
    plan_path = str(plan.get("_plan_path", ""))
    if not expected_commit or not plan_path or not Path(plan_path).is_file():
        return False

    sources: dict[str, tuple[str, str]] = {
        "plan": (plan_path, sha256_file(Path(plan_path))),
    }
    for spec in plan.get("cells", []):
        label = str(spec.get("cell", ""))
        if not label:
            return False
        bindings = {
            f"scored_target:{label}": spec.get("provenance", {})
            .get("test", {})
            .get("scored_target"),
            f"trajuq_predictions:{label}": spec.get("trajuq_predictions"),
            f"repeated_metrics:{label}": spec.get("repeated_metrics"),
            f"repeated_predictions:{label}": spec.get("repeated_predictions"),
        }
        for name, raw_path in bindings.items():
            path = Path(str(raw_path or ""))
            if not path.is_file():
                return False
            sources[name] = (str(path), sha256_file(path))

    return binding_matches(
        payload.get("provenance", {}),
        expected_commit=expected_commit,
        expected_sources=sources,
    )


def macro_state(plan: dict[str, Any]) -> str:
    path = Path(str(plan["macro_output"]))
    if not path.exists():
        return "MISSING"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "INVALID"
    expected_cells = int(plan.get("expected_cells", 0))
    compute = payload.get("auxiliary_compute", {})
    try:
        mean_compute = float(
            compute.get(
                "mean_test_auxiliary_wall_time_seconds_per_example",
                "nan",
            )
        )
        mean_p90 = float(
            compute.get(
                "mean_test_p90_auxiliary_wall_time_seconds_per_example",
                "nan",
            )
        )
    except (TypeError, ValueError):
        mean_compute = float("nan")
        mean_p90 = float("nan")

    complete = (
        payload.get("protocol") == "trajuq_vs_repeated_macro_v2"
        and payload.get("comparison")
        == "single_trajectory_trajuq_vs_five_sample_baseline"
        and payload.get("cell_set_complete") is True
        and int(payload.get("expected_cells", -1)) == expected_cells
        and int(payload.get("observed_cells", -1)) == expected_cells
        and compute.get("aggregation") == "equal_cell_weight_macro"
        and compute.get("complete") is True
        and int(compute.get("expected_cells", -1)) == expected_cells
        and int(
            compute.get("observed_cells_with_complete_test_wall_time", -1)
        )
        == expected_cells
        and int(compute.get("sample_count", -1)) == 5
        and mean_compute >= 0.0
        and mean_compute < float("inf")
        and mean_p90 >= 0.0
        and mean_p90 < float("inf")
        and repeated_macro_provenance_matches(plan, payload)
        and set(payload.get("macro", {})) >= {
            "auroc",
            "auprc",
            "brier",
            "nll",
            "ece15",
            "aurc",
            "risk50",
            "risk80",
        }
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
    plan["_plan_path"] = str(plan_path)
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

        try:
            expected_test_ids = artifact_example_ids(
                Path(str(cell["provenance"]["test"]["scored_target"]))
            )
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise SystemExit(
                f"{scope}: cannot resolve frozen repeated test IDs: {exc}"
            ) from exc

        evaluation = pair_state(
            Path(str(cell["repeated_metrics"])),
            Path(str(cell["repeated_predictions"])),
            expected_test_ids,
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
                expected_test_ids,
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

        comparison = comparison_state(
            Path(str(cell["direct_comparison"])),
            len(expected_test_ids),
        )
        if comparison == "MISSING":
            run_command(
                [str(value) for value in cell["compare_to_trajuq_argv"]],
                execute=args.execute,
                scope=scope,
                step="compare_to_trajuq",
                event_log=event_path,
            )
            if args.execute and comparison_state(
                Path(str(cell["direct_comparison"])),
                len(expected_test_ids),
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
