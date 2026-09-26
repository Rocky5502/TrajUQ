#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

from trajuq.features import FINAL_BASELINE_FEATURES, PRIMARY_TRAJECTORY_FEATURES
from trajuq.io import read_jsonl
from trajuq.schema import SCHEMA_VERSION


BOUNDED_FEATURES = {
    "entropy_mean",
    "entropy_p90",
    "entropy_max",
    "entropy_final",
    "confidence_mean",
    "confidence_final",
    "margin_mean",
    "margin_min",
    "margin_final",
    "decision_entropy_mean",
    "decision_entropy_p90",
    "decision_confidence_mean",
    "decision_margin_mean",
    "js_mean",
    "js_p90",
    "js_max",
    "js_final",
    "revision_rate",
    "late_revision_rate",
    "mask_auc",
    "mask_fraction_final",
    "late_mask_fraction",
    "commitment_time_mean",
    "commitment_time_p90",
}

BOUNDED_STEP_FIELDS = {
    "entropy_mean",
    "entropy_p90",
    "confidence_mean",
    "margin_mean",
    "mask_fraction",
    "revision_fraction",
    "js_mean",
    "active_fraction",
}

FORBIDDEN_RAW_METADATA_KEYS = {
    "reference",
    "correct",
    "gold",
    "gold_answer",
    "answer_key",
    "label",
    "target",
}


def recursive_keys(value) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(recursive_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(recursive_keys(child))
    return keys


def audit_run_provenance(
    rows: list[dict],
    *,
    require_revision: bool,
) -> tuple[list[str], dict[str, list]]:
    failures: list[str] = []
    datasets: set[str] = set()
    models: set[str] = set()
    adapters: set[str] = set()
    seeds: set[int] = set()
    revisions: set[str] = set()

    for i, row in enumerate(rows):
        for field, values in (
            ("dataset", datasets),
            ("model", models),
            ("adapter", adapters),
        ):
            value = row.get(field)
            text = "" if value is None else str(value).strip()
            if not text:
                failures.append(f"row {i}: missing {field}")
            else:
                values.add(text)

        prompt_hash = row.get("prompt_sha256")
        if (
            not isinstance(prompt_hash, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", prompt_hash) is None
        ):
            failures.append(
                f"row {i}: prompt_sha256 must be a 64-character hexadecimal digest"
            )

        seed = row.get("seed")
        if isinstance(seed, bool):
            failures.append(f"row {i}: seed must be integer-like")
        else:
            try:
                seed_value = int(seed)
            except (TypeError, ValueError):
                failures.append(f"row {i}: seed must be integer-like")
            else:
                seeds.add(seed_value)

        record_revision = row.get("metadata", {}).get("model_revision")
        generation_revision = (
            row.get("generation", {})
            .get("metadata", {})
            .get("model_revision")
        )
        if require_revision and not record_revision:
            failures.append(
                f"row {i}: record metadata model revision is not pinned"
            )
        if require_revision and not generation_revision:
            failures.append(
                f"row {i}: generation metadata model revision is not pinned"
            )
        if (
            record_revision
            and generation_revision
            and str(record_revision) != str(generation_revision)
        ):
            failures.append(
                f"row {i}: record/generation model revisions disagree"
            )
        revision = generation_revision or record_revision
        if revision:
            revisions.add(str(revision))

    for label, values in (
        ("dataset", datasets),
        ("model", models),
        ("adapter", adapters),
        ("seed", seeds),
    ):
        if rows and len(values) != 1:
            failures.append(
                f"run must contain exactly one {label}; observed={sorted(values)}"
            )
    if len(revisions) > 1:
        failures.append(
            "run must contain exactly one model revision; "
            f"observed={sorted(revisions)}"
        )
    if rows and require_revision and len(revisions) != 1:
        failures.append(
            "run requires exactly one pinned model revision"
        )

    return failures, {
        "datasets": sorted(datasets),
        "datasets": provenance["datasets"],
        "models": provenance["models"],
        "adapters": provenance["adapters"],
        "seeds": provenance["seeds"],
        "model_revisions": provenance["model_revisions"],
        "seeds": sorted(seeds),
        "model_revisions": sorted(revisions),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def git_clean() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return status.strip() == ""
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw or scored TrajUQ JSONL artifacts.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--stage", choices=["raw", "scored", "auto"], default="auto")
    parser.add_argument("--require-revision", action="store_true")
    args = parser.parse_args()

    path = Path(args.run)
    rows = list(read_jsonl(path))
    ids: set[str] = set()
    failures: list[str] = []
    feature_names: set[str] | None = None
    schema_versions: set[str] = set()

    if not rows:
        failures.append("artifact contains zero records")

    if args.stage == "auto":
        has_labels = [r.get("correct") is not None for r in rows]
        stage = "scored" if rows and all(has_labels) else "raw"
        if any(has_labels) and not all(has_labels):
            failures.append("mixed scored/unscored rows")
    else:
        stage = args.stage

    provenance_failures, provenance = audit_run_provenance(
        rows,
        require_revision=args.require_revision,
    )
    failures.extend(provenance_failures)

    for i, row in enumerate(rows):
        item_id = str(row.get("example_id", ""))
        if not item_id:
            failures.append(f"row {i}: missing example_id")
        elif item_id in ids:
            failures.append(f"row {i}: duplicate example_id={item_id}")
        ids.add(item_id)

        schema_version = row.get("schema_version")
        if not schema_version:
            failures.append(f"row {i}: missing schema_version")
        else:
            schema_versions.add(str(schema_version))
            if str(schema_version) != SCHEMA_VERSION:
                failures.append(
                    f"row {i}: schema_version={schema_version!r}; expected {SCHEMA_VERSION!r}"
                )

        feats = row.get("generation", {}).get("features", {})
        names = set(feats)
        if not feats:
            failures.append(f"row {i}: missing trajectory features")
        else:
            required = set(FINAL_BASELINE_FEATURES) | set(PRIMARY_TRAJECTORY_FEATURES)
            missing_required = sorted(required - set(feats))
            if missing_required:
                failures.append(
                    f"row {i}: missing frozen primary features={missing_required}"
                )
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            failures.append(f"row {i}: feature schema mismatch")

        for key, value in feats.items():
            try:
                number = float(value)
                finite = math.isfinite(number)
            except (TypeError, ValueError):
                number = float("nan")
                finite = False
            if not finite:
                failures.append(f"row {i}: non-finite feature {key}={value!r}")
                continue
            if key in BOUNDED_FEATURES and not (-1e-6 <= number <= 1.0 + 1e-6):
                failures.append(
                    f"row {i}: bounded feature {key} outside [0,1]: {number}"
                )

        trajectory_steps = feats.get("trajectory_steps")
        if trajectory_steps is not None:
            try:
                steps_value = float(trajectory_steps)
            except (TypeError, ValueError):
                steps_value = float("nan")
            if (
                not math.isfinite(steps_value)
                or steps_value <= 0
                or abs(steps_value - round(steps_value)) > 1e-6
            ):
                failures.append(
                    f"row {i}: trajectory_steps must be a positive integer-like value"
                )

        step_rows = row.get("generation", {}).get("steps", [])
        if not isinstance(step_rows, list) or not step_rows:
            failures.append(f"row {i}: missing per-step trajectory summaries")
        else:
            for step_j, summary in enumerate(step_rows):
                if not isinstance(summary, dict):
                    failures.append(f"row {i} step {step_j}: summary is not an object")
                    continue
                for field in BOUNDED_STEP_FIELDS:
                    if field not in summary:
                        continue
                    try:
                        number = float(summary[field])
                    except (TypeError, ValueError):
                        failures.append(
                            f"row {i} step {step_j}: non-numeric {field}"
                        )
                        continue
                    if not math.isfinite(number) or not (-1e-6 <= number <= 1.0 + 1e-6):
                        failures.append(
                            f"row {i} step {step_j}: {field} outside [0,1]: {number}"
                        )
                active_count = summary.get("active_count")
                if active_count is not None:
                    try:
                        active_number = int(active_count)
                    except (TypeError, ValueError):
                        failures.append(
                            f"row {i} step {step_j}: active_count is not integer-like"
                        )
                    else:
                        if active_number < 0:
                            failures.append(
                                f"row {i} step {step_j}: active_count is negative"
                            )

        if row.get("reference") is not None:
            failures.append(f"row {i}: reference must never be stored in run artifacts")

        if stage == "raw":
            if row.get("correct") is not None:
                failures.append(f"row {i}: correctness leaked into raw artifact")
            metadata_keys = recursive_keys(
                {
                    "record_metadata": row.get("metadata", {}),
                    "generation_metadata": row.get("generation", {}).get("metadata", {}),
                }
            )
            leaked_keys = sorted(metadata_keys & FORBIDDEN_RAW_METADATA_KEYS)
            if leaked_keys:
                failures.append(
                    f"row {i}: outcome-like metadata keys leaked into raw artifact: "
                    f"{leaked_keys}"
                )
        if stage == "scored" and not isinstance(row.get("correct"), bool):
            failures.append(f"row {i}: scored artifact requires boolean correctness")

        adapter_name = str(row.get("adapter", ""))
        if adapter_name in {"llada", "dream", "sdar"}:
            semantics = row.get("generation", {}).get("metadata", {}).get(
                "trajectory_semantics"
            )
            if semantics != "active_unresolved_positions_v4":
                failures.append(
                    f"row {i}: trajectory_semantics={semantics!r}; "
                    "expected active_unresolved_positions_v4"
                )

    commit = git_head()
    clean = git_clean()

    audit = {
        "protocol": "trajuq_run_audit_v1",
        "git_commit": commit,
        "git_clean": clean,
        "file": str(path),
        "stage": stage,
        "sha256": sha256_file(path),
        "records": len(rows),
        "unique_ids": len(ids),
        "models": sorted(models),
        "adapters": sorted(adapters),
        "schema_versions": sorted(schema_versions),
        "expected_schema_version": SCHEMA_VERSION,
        "feature_count": len(feature_names or set()),
        "feature_names": sorted(feature_names or set()),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }

    out = Path(args.output) if args.output else path.with_suffix(path.suffix + ".audit.json")
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
