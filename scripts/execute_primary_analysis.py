#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_rq1_claim_gate import evaluate_gate  # noqa: E402
from scripts.execute_primary_plan import (  # noqa: E402
    validate_plan_provenance as validate_primary_plan_provenance,
)
from trajuq.features import FROZEN_FEATURE_GROUPS  # noqa: E402
from trajuq.io import read_jsonl  # noqa: E402
from trajuq.provenance import binding_matches, compact_binding_matches  # noqa: E402


PRIOR_SIGNAL_NAMES = {"trajectory_entropy", "top1_revision_rate"}


CELL_STEPS = (
    ("evaluate", "evaluate_argv", ("metrics", "predictions")),
    ("nested_comparison", "nested_comparison_argv", ("nested_bootstrap",)),
    (
        "feature_ablation",
        "feature_ablation_argv",
        ("feature_ablation", "feature_ablation_predictions"),
    ),
    (
        "prior_signal_diagnostic",
        "prior_signal_diagnostic_argv",
        ("prior_signal_diagnostic", "prior_signal_predictions"),
    ),
    ("raw_risk_diagnostic", "raw_risk_diagnostic_argv", ("raw_risk_diagnostic",)),
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


def artifact_example_ids(path: Path) -> list[str]:
    rows = list(read_jsonl(path))
    ids = [str(row.get("example_id", "")) for row in rows]
    if not rows:
        raise ValueError(f"artifact is empty: {path}")
    if any(not item_id for item_id in ids):
        raise ValueError(f"artifact has a row without example_id: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"artifact has duplicate example_ids: {path}")
    return ids


def expected_partition_ids(cell: dict[str, Any], partition: str) -> list[str]:
    path = Path(str(cell["inputs"][partition]["path"]))
    return artifact_example_ids(path)


def expected_analysis_sources(
    cell: dict[str, Any],
    *,
    include_calibration: bool = True,
) -> dict[str, tuple[str, str]]:
    sources: dict[str, tuple[str, str]] = {}
    if include_calibration:
        cal = cell["inputs"]["calibration"]
        sources["calibration"] = (
            str(cal["path"]),
            str(cal["sha256"]),
        )
    test = cell["inputs"]["test"]
    sources["test"] = (
        str(test["path"]),
        str(test["sha256"]),
    )
    return sources


def json_provenance_matches(
    cell: dict[str, Any],
    payload: dict,
    *,
    include_calibration: bool = True,
) -> bool:
    expected_commit = str(cell.get("experiment_git_commit", ""))
    if not expected_commit:
        return False
    return binding_matches(
        payload.get("provenance", {}),
        expected_commit=expected_commit,
        expected_sources=expected_analysis_sources(
            cell,
            include_calibration=include_calibration,
        ),
    )


def row_provenance_matches(
    cell: dict[str, Any],
    row: dict,
    *,
    include_calibration: bool = True,
) -> bool:
    expected_commit = str(cell.get("experiment_git_commit", ""))
    if not expected_commit:
        return False
    expected_sources = expected_analysis_sources(
        cell,
        include_calibration=include_calibration,
    )
    return compact_binding_matches(
        row.get("analysis_provenance", {}),
        expected_commit=expected_commit,
        expected_sha256={
            name: digest
            for name, (_, digest) in expected_sources.items()
        },
    )


def semantic_output_valid(
    cell: dict[str, Any],
    key: str,
    path: Path,
) -> bool:
    if not output_valid(path):
        return False

    try:
        n_cal = len(expected_partition_ids(cell, "calibration"))
        test_ids = expected_partition_ids(cell, "test")
    except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError):
        return False
    n_test = len(test_ids)

    if key == "predictions":
        try:
            rows = list(read_jsonl(path))
            return (
                artifact_example_ids(path) == test_ids
                and all(row_provenance_matches(cell, row) for row in rows)
            )
        except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError):
            return False

    if key == "feature_ablation_predictions":
        try:
            rows = list(read_jsonl(path))
            if artifact_example_ids(path) != test_ids:
                return False
            expected_groups = set(FROZEN_FEATURE_GROUPS)
            target_rows = list(read_jsonl(Path(str(cell["inputs"]["test"]["path"]))))
            target_by_id = {str(row["example_id"]): row for row in target_rows}
            for row in rows:
                if not row_provenance_matches(cell, row):
                    return False
                item_id = str(row["example_id"])
                target = target_by_id[item_id]
                expected_y = 0 if bool(target["correct"]) else 1
                if int(row.get("y_error", -1)) != expected_y:
                    return False
                if bool(row.get("correct")) != bool(target["correct"]):
                    return False
                scores = row.get("p_error_by_group", {})
                if set(scores) != expected_groups:
                    return False
                for value in scores.values():
                    number = float(value)
                    if not (0.0 <= number <= 1.0):
                        return False
            return True
        except (
            OSError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
            KeyError,
        ):
            return False

    if key == "prior_signal_predictions":
        try:
            rows = list(read_jsonl(path))
            if artifact_example_ids(path) != test_ids:
                return False
            target_rows = list(
                read_jsonl(Path(str(cell["inputs"]["test"]["path"])))
            )
            target_by_id = {
                str(row["example_id"]): row for row in target_rows
            }
            fields = {
                "p_error_trajectory_entropy",
                "p_error_top1_revision_rate",
            }
            for row in rows:
                if not row_provenance_matches(cell, row):
                    return False
                item_id = str(row["example_id"])
                target = target_by_id[item_id]
                expected_y = 0 if bool(target["correct"]) else 1
                if int(row.get("y_error", -1)) != expected_y:
                    return False
                if bool(row.get("correct")) != bool(target["correct"]):
                    return False
                for field in fields:
                    number = float(row[field])
                    if not (0.0 <= number <= 1.0):
                        return False
            return True
        except (
            OSError,
            json.JSONDecodeError,
            ValueError,
            TypeError,
            KeyError,
        ):
            return False

    if path.suffix != ".json":
        return output_valid(path)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if key == "metrics":
        return (
            payload.get("protocol") == "trajuq_heldout_evaluation_v1"
            and json_provenance_matches(cell, payload)
            and int(
                payload.get("decision_entropy_baseline", {}).get("n_calibration", -1)
            )
            == n_cal
            and int(
                payload.get("decision_entropy_baseline", {}).get("n_test", -1)
            )
            == n_test
            and int(
                payload.get("trajectory_plus_decision_logistic", {}).get(
                    "n_calibration",
                    -1,
                )
            )
            == n_cal
            and int(
                payload.get("trajectory_plus_decision_logistic", {}).get(
                    "n_test",
                    -1,
                )
            )
            == n_test
        )
    if key == "nested_bootstrap":
        return (
            payload.get("protocol")
            == "trajuq_nested_calibrated_comparison_v1"
            and json_provenance_matches(cell, payload)
            and int(payload.get("n_calibration", -1)) == n_cal
            and int(payload.get("n_test", -1)) == n_test
            and set(payload.get("metrics", {})) >= {"auroc", "auprc", "aurc"}
        )
    if key == "feature_ablation":
        expected_groups = list(FROZEN_FEATURE_GROUPS)
        return (
            payload.get("protocol") == "frozen_feature_family_ablation_v1"
            and json_provenance_matches(cell, payload)
            and int(payload.get("n_calibration", -1)) == n_cal
            and int(payload.get("n_test", -1)) == n_test
            and payload.get("baseline") == "decision_only"
            and payload.get("group_order") == expected_groups
            and set(payload.get("groups", {})) == set(expected_groups)
            and set(payload.get("paired_improvements_vs_decision_only", {}))
            == (set(expected_groups) - {"decision_only"})
        )
    if key == "prior_signal_diagnostic":
        signals = payload.get("signals", {})
        return (
            payload.get("protocol")
            == "trajuq_prior_aligned_signal_diagnostic_v1"
            and json_provenance_matches(cell, payload)
            and payload.get("role")
            == "secondary_prior_aligned_signal_class_diagnostic"
            and int(payload.get("n_calibration", -1)) == n_cal
            and int(payload.get("n_test", -1)) == n_test
            and payload.get("signal_order")
            == ["trajectory_entropy", "top1_revision_rate"]
            and set(signals) == PRIOR_SIGNAL_NAMES
            and all(
                int(signals[name].get("n_calibration", -1)) == n_cal
                and int(signals[name].get("n_test", -1)) == n_test
                for name in PRIOR_SIGNAL_NAMES
            )
        )
    if key == "raw_risk_diagnostic":
        return (
            payload.get("protocol") == "trajuq_raw_risk_diagnostic_v1"
            and json_provenance_matches(
                cell,
                payload,
                include_calibration=False,
            )
            and int(payload.get("n", -1)) == n_test
            and set(payload.get("metrics", {})) >= {"auroc", "auprc", "aurc"}
        )
    return True


def validate_plan_provenance(
    plan: dict[str, Any],
    *,
    require_current_commit: bool = True,
) -> list[str]:
    failures: list[str] = []
    if plan.get("protocol") != "trajuq_primary_analysis_plan_v2":
        failures.append("primary analysis plan protocol is not v2")

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
            failures.append(f"analysis plan missing bound {key} artifact")
        elif sha256_file(path) != expected_hash:
            failures.append(f"{key} hash differs from analysis plan")

    primary_plan_path = Path(str(plan.get("primary_plan", "")))
    if primary_plan_path.is_file():
        try:
            primary_plan = json.loads(
                primary_plan_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"bound primary plan is invalid JSON: {exc}")
        else:
            failures.extend(
                "primary plan provenance: " + failure
                for failure in validate_primary_plan_provenance(
                    primary_plan,
                    require_current_commit=require_current_commit,
                )
            )

    for cell in plan.get("cell_analyses", []):
        label = str(cell.get("cell", "<unknown>"))
        inputs = cell.get("inputs", {})
        for partition in ("calibration", "test"):
            spec = inputs.get(partition, {})

            source_input = Path(str(spec.get("source_input", "")))
            expected_source_hash = str(spec.get("source_input_sha256", ""))
            raw = Path(str(spec.get("raw_path", "")))
            expected_raw_hash = str(spec.get("raw_sha256", ""))
            raw_audit = Path(str(spec.get("raw_audit_path", "")))
            expected_raw_audit_hash = str(spec.get("raw_audit_sha256", ""))

            for bound_name, path, expected_hash in (
                ("source input", source_input, expected_source_hash),
                ("raw input", raw, expected_raw_hash),
                ("raw audit", raw_audit, expected_raw_audit_hash),
            ):
                if not path.exists() or not expected_hash:
                    failures.append(
                        f"{label} {partition}: bound {bound_name} missing"
                    )
                    continue
                if sha256_file(path) != expected_hash:
                    failures.append(
                        f"{label} {partition}: {bound_name} hash differs from plan"
                    )

            if raw_audit.exists():
                try:
                    raw_audit_payload = json.loads(
                        raw_audit.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    failures.append(
                        f"{label} {partition}: raw audit is invalid JSON"
                    )
                else:
                    if (
                        raw_audit_payload.get("protocol")
                        != "trajuq_run_audit_v1"
                        or raw_audit_payload.get("status") != "PASS"
                        or raw_audit_payload.get("git_clean") is not True
                        or raw_audit_payload.get("git_commit") != expected_commit
                        or raw_audit_payload.get("stage") != "raw"
                        or str(raw_audit_payload.get("sha256", ""))
                        != expected_raw_hash
                    ):
                        failures.append(
                            f"{label} {partition}: raw audit is not a "
                            "clean frozen hash-bound PASS"
                        )

            scored = Path(str(spec.get("path", "")))
            expected_scored_hash = str(spec.get("sha256", ""))
            audit = Path(str(spec.get("audit_path", "")))
            expected_audit_hash = str(spec.get("audit_sha256", ""))

            if not scored.exists() or not expected_scored_hash:
                failures.append(f"{label} {partition}: bound scored input missing")
                continue
            observed_scored_hash = sha256_file(scored)
            if observed_scored_hash != expected_scored_hash:
                failures.append(
                    f"{label} {partition}: scored input hash differs from plan"
                )

            if not audit.exists() or not expected_audit_hash:
                failures.append(f"{label} {partition}: bound audit missing")
                continue
            if sha256_file(audit) != expected_audit_hash:
                failures.append(
                    f"{label} {partition}: audit hash differs from plan"
                )
                continue
            try:
                audit_payload = json.loads(audit.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                failures.append(f"{label} {partition}: audit is invalid JSON")
                continue
            if (
                audit_payload.get("protocol") != "trajuq_run_audit_v1"
                or audit_payload.get("status") != "PASS"
                or audit_payload.get("git_clean") is not True
                or audit_payload.get("git_commit") != expected_commit
                or audit_payload.get("stage") != "scored"
            ):
                failures.append(
                    f"{label} {partition}: audit is not a clean frozen scored PASS"
                )
            if str(audit_payload.get("sha256", "")) != observed_scored_hash:
                failures.append(
                    f"{label} {partition}: audit no longer binds scored input"
                )

    return failures


def step_state(cell: dict[str, Any], output_keys: tuple[str, ...]) -> str:
    outputs = cell["outputs"]
    paths = [Path(str(outputs[key])) for key in output_keys]
    exists = [path.exists() for path in paths]
    if not any(exists):
        return "MISSING"
    if not all(exists):
        return "PARTIAL"
    if all(
        semantic_output_valid(cell, key, path)
        for key, path in zip(output_keys, paths)
    ):
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


def aggregate_provenance_matches(
    plan: dict[str, Any],
    payload: dict,
    kind: str,
) -> bool:
    # Lightweight unit fixtures may omit the real plan protocol. Real frozen
    # analysis plans must carry source-bound aggregate artifacts.
    if plan.get("protocol") != "trajuq_primary_analysis_plan_v2":
        return True

    expected_commit = str(plan.get("experiment_git_commit", ""))
    if not expected_commit:
        return False

    sources: dict[str, tuple[str, str]] = {}
    cells = list(plan.get("cell_analyses", []))

    if kind == "primary":
        config_path = str(plan.get("primary_config", ""))
        config_hash = str(plan.get("primary_config_sha256", ""))
        if not config_path or not config_hash:
            return False
        sources["primary_config"] = (config_path, config_hash)
        for cell in cells:
            label = str(cell.get("cell", ""))
            path = Path(str(cell.get("outputs", {}).get("predictions", "")))
            if not label or not path.is_file():
                return False
            sources[f"prediction:{label.replace(':', '_')}"] = (
                str(path),
                sha256_file(path),
            )
    elif kind == "ablation":
        for cell in cells:
            label = str(cell.get("cell", ""))
            path = Path(
                str(
                    cell.get("outputs", {}).get(
                        "feature_ablation_predictions",
                        "",
                    )
                )
            )
            if not label or not path.is_file():
                return False
            sources[f"prediction:{label}"] = (
                str(path),
                sha256_file(path),
            )
    elif kind == "prior":
        for cell in cells:
            label = str(cell.get("cell", ""))
            outputs = cell.get("outputs", {})
            trajuq_path = Path(str(outputs.get("predictions", "")))
            prior_path = Path(str(outputs.get("prior_signal_predictions", "")))
            if (
                not label
                or not trajuq_path.is_file()
                or not prior_path.is_file()
            ):
                return False
            sources[f"trajuq_prediction:{label}"] = (
                str(trajuq_path),
                sha256_file(trajuq_path),
            )
            sources[f"prior_signal_prediction:{label}"] = (
                str(prior_path),
                sha256_file(prior_path),
            )
    elif kind == "claim":
        macro_path = Path(
            str(plan.get("final_outputs", {}).get("macro_primary", ""))
        )
        config_path = str(plan.get("primary_config", ""))
        config_hash = str(plan.get("primary_config_sha256", ""))
        if not macro_path.is_file() or not config_path or not config_hash:
            return False
        sources = {
            "macro_primary": (str(macro_path), sha256_file(macro_path)),
            "primary_config": (config_path, config_hash),
        }
    else:
        raise ValueError(f"unknown aggregate provenance kind: {kind}")

    return binding_matches(
        payload.get("provenance", {}),
        expected_commit=expected_commit,
        expected_sources=sources,
    )


def claim_matches_frozen_rule(
    plan: dict[str, Any],
    payload: dict[str, Any],
) -> bool:
    """Re-evaluate the RQ1 gate so a hand-edited decision cannot be reused."""

    # Preserve lightweight unit fixtures that intentionally omit the full plan.
    if plan.get("protocol") != "trajuq_primary_analysis_plan_v2":
        return True

    macro_path = Path(
        str(plan.get("final_outputs", {}).get("macro_primary", ""))
    )
    config_path = Path(str(plan.get("primary_config", "")))
    if not macro_path.is_file() or not config_path.is_file():
        return False
    try:
        macro_payload = json.loads(macro_path.read_text(encoding="utf-8"))
        primary_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        expected = evaluate_gate(macro_payload, primary_cfg)
    except (OSError, json.JSONDecodeError, yaml.YAMLError, KeyError, TypeError, ValueError):
        return False

    for key in (
        "primary_endpoint",
        "decision",
        "claim_supported",
        "frozen_rules",
        "observed",
        "conditions",
    ):
        if payload.get(key) != expected.get(key):
            return False
    return True


def macro_state(plan: dict[str, Any]) -> dict[str, Any]:
    macro = Path(str(plan["final_outputs"]["macro_primary"]))
    ablation = Path(str(plan["final_outputs"]["macro_feature_ablation"]))
    prior = Path(
        str(plan["final_outputs"]["macro_prior_signal_diagnostics"])
    )
    claim = Path(str(plan["final_outputs"]["rq1_claim_gate"]))
    expected_cells = int(plan.get("cell_count", 0))

    macro_status = "MISSING"
    if macro.exists():
        if not valid_json(macro):
            macro_status = "INVALID"
        else:
            payload = json.loads(macro.read_text(encoding="utf-8"))
            macro_status = (
                "COMPLETE"
                if (
                    payload.get("protocol")
                    == "hierarchical_within_cell_macro_bootstrap_v1"
                    and aggregate_provenance_matches(
                        plan,
                        payload,
                        "primary",
                    )
                    and payload.get("cell_set_complete") is True
                    and int(payload.get("observed_cell_count", -1))
                    == expected_cells
                    and int(payload.get("expected_cell_count", -1))
                    == expected_cells
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
                else "INVALID"
            )

    ablation_status = "MISSING"
    if ablation.exists():
        if not valid_json(ablation):
            ablation_status = "INVALID"
        else:
            payload = json.loads(ablation.read_text(encoding="utf-8"))
            expected_groups = set(FROZEN_FEATURE_GROUPS) - {"decision_only"}
            groups = payload.get("groups", {})
            metrics_complete = all(
                set(groups.get(group, {}).get("metrics", {}))
                >= {"auroc", "auprc", "aurc"}
                for group in expected_groups
            )
            ablation_status = (
                "COMPLETE"
                if (
                    payload.get("protocol")
                    == "trajuq_feature_ablation_macro_v1"
                    and aggregate_provenance_matches(
                        plan,
                        payload,
                        "ablation",
                    )
                    and payload.get("baseline") == "decision_only"
                    and payload.get("cell_set_complete") is True
                    and int(payload.get("observed_cells", -1))
                    == expected_cells
                    and int(payload.get("expected_cells", -1))
                    == expected_cells
                    and set(groups) == expected_groups
                    and metrics_complete
                )
                else "INVALID"
            )

    prior_status = "MISSING"
    if prior.exists():
        if not valid_json(prior):
            prior_status = "INVALID"
        else:
            payload = json.loads(prior.read_text(encoding="utf-8"))
            macro = payload.get("macro", {})
            metrics_complete = all(
                set(macro.get(signal, {})) >= {
                    "auroc",
                    "auprc",
                    "aurc",
                }
                for signal in PRIOR_SIGNAL_NAMES
            )
            prior_status = (
                "COMPLETE"
                if (
                    payload.get("protocol")
                    == "trajuq_prior_aligned_signal_macro_v1"
                    and aggregate_provenance_matches(
                        plan,
                        payload,
                        "prior",
                    )
                    and payload.get("role")
                    == "secondary_prior_aligned_signal_class_diagnostic"
                    and payload.get("cell_set_complete") is True
                    and int(payload.get("observed_cells", -1))
                    == expected_cells
                    and int(payload.get("expected_cells", -1))
                    == expected_cells
                    and set(macro) == PRIOR_SIGNAL_NAMES
                    and metrics_complete
                )
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
                if (
                    payload.get("protocol") == "trajuq_rq1_claim_gate_v1"
                    and aggregate_provenance_matches(
                        plan,
                        payload,
                        "claim",
                    )
                    and payload.get("decision")
                    in {
                        "SUPPORT_POSITIVE_RQ1_CLAIM",
                        "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM",
                    }
                    and isinstance(payload.get("claim_supported"), bool)
                    and isinstance(payload.get("conditions"), dict)
                    and claim_matches_frozen_rule(plan, payload)
                )
                else "INVALID"
            )

    return {
        "macro_primary": macro_status,
        "macro_feature_ablation": ablation_status,
        "macro_prior_signal_diagnostics": prior_status,
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
        and final["macro_feature_ablation"] == "COMPLETE"
        and final["macro_prior_signal_diagnostics"] == "COMPLETE"
        and final["rq1_claim_gate"] == "COMPLETE"
        else "IN_PROGRESS"
    )
    return {
        "protocol": "trajuq_primary_analysis_execution_state_v2",
        "status": status,
        "plan_protocol": plan.get("protocol"),
        "experiment_git_commit": plan.get("experiment_git_commit"),
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

    provenance_failures = validate_plan_provenance(plan)
    if provenance_failures:
        raise SystemExit(
            "Primary analysis provenance validation failed:\n- "
            + "\n- ".join(provenance_failures)
        )

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
            if (
                args.execute
                and macro_state(plan)["macro_primary"] != "COMPLETE"
            ):
                write_snapshot(plan, state_path)
                raise SystemExit(
                    "Primary macro command did not produce a complete macro."
                )
        elif final["macro_primary"] == "INVALID":
            raise SystemExit(
                "Existing primary macro is invalid; "
                "do not overwrite it automatically."
            )
        else:
            print("[SKIP] primary macro_aggregation: complete")

        final = macro_state(plan)
        if final["macro_primary"] == "COMPLETE":
            if final["macro_feature_ablation"] == "MISSING":
                run_command(
                    [
                        str(value)
                        for value in plan["feature_ablation_macro_argv"]
                    ],
                    execute=args.execute,
                    scope="primary",
                    step="feature_ablation_macro",
                    event_log=event_path,
                )
                if (
                    args.execute
                    and macro_state(plan)["macro_feature_ablation"]
                    != "COMPLETE"
                ):
                    write_snapshot(plan, state_path)
                    raise SystemExit(
                        "Feature-ablation macro command did not produce "
                        "a complete artifact."
                    )
            elif final["macro_feature_ablation"] == "INVALID":
                raise SystemExit(
                    "Existing feature-ablation macro is invalid; "
                    "do not overwrite it automatically."
                )
            else:
                print("[SKIP] primary feature_ablation_macro: complete")

        final = macro_state(plan)
        if (
            final["macro_primary"] == "COMPLETE"
            and final["macro_feature_ablation"] == "COMPLETE"
        ):
            if final["macro_prior_signal_diagnostics"] == "MISSING":
                run_command(
                    [
                        str(value)
                        for value in plan["prior_signal_macro_argv"]
                    ],
                    execute=args.execute,
                    scope="primary",
                    step="prior_signal_macro",
                    event_log=event_path,
                )
                if (
                    args.execute
                    and macro_state(plan)[
                        "macro_prior_signal_diagnostics"
                    ]
                    != "COMPLETE"
                ):
                    write_snapshot(plan, state_path)
                    raise SystemExit(
                        "Prior-signal macro command did not produce "
                        "a complete artifact."
                    )
            elif (
                final["macro_prior_signal_diagnostics"] == "INVALID"
            ):
                raise SystemExit(
                    "Existing prior-signal macro is invalid; "
                    "do not overwrite it automatically."
                )
            else:
                print("[SKIP] primary prior_signal_macro: complete")

        final = macro_state(plan)
        if (
            final["macro_primary"] == "COMPLETE"
            and final["macro_feature_ablation"] == "COMPLETE"
            and final["macro_prior_signal_diagnostics"] == "COMPLETE"
        ):
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
        f"{final_snapshot['complete_cells']}/"
        f"{final_snapshot['expected_cells']} cells; "
        f"macro={final_snapshot['final']['macro_primary']}; "
        f"ablation={final_snapshot['final']['macro_feature_ablation']}; "
        f"prior={final_snapshot['final']['macro_prior_signal_diagnostics']}; "
        f"claim_gate={final_snapshot['final']['rq1_claim_gate']}"
    )


if __name__ == "__main__":
    main()
