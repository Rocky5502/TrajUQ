#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from trajuq.features import PRIMARY_TRAJECTORY_FEATURES
from trajuq.io import read_jsonl
from trajuq.risk import raw_trajectory_risk
from trajuq.schema import SCHEMA_VERSION


def load_unique(path: str) -> tuple[list[dict], dict[str, dict]]:
    rows = list(read_jsonl(path))
    index: dict[str, dict] = {}
    for row in rows:
        item_id = str(row.get("example_id", ""))
        if not item_id:
            raise SystemExit(f"{path}: missing example_id")
        if item_id in index:
            raise SystemExit(f"{path}: duplicate example_id={item_id}")
        index[item_id] = row
    return rows, index


def finite_float(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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
    parser = argparse.ArgumentParser(description="Automated 50-item TrajUQ pilot quality gate.")
    parser.add_argument("--raw", required=True)
    parser.add_argument("--scored", required=True)
    parser.add_argument("--output", default="results/pilot_quality_gate.json")
    parser.add_argument("--expected-records", type=int, default=50)
    parser.add_argument("--min-varying-primary-features", type=int, default=5)
    parser.add_argument("--std-eps", type=float, default=1e-8)
    args = parser.parse_args()

    raw_rows, raw = load_unique(args.raw)
    scored_rows, scored = load_unique(args.scored)
    failures: list[str] = []
    warnings: list[str] = []

    if len(raw_rows) != args.expected_records:
        failures.append(f"raw record count={len(raw_rows)}; expected {args.expected_records}")
    if len(scored_rows) != args.expected_records:
        failures.append(f"scored record count={len(scored_rows)}; expected {args.expected_records}")
    if set(raw) != set(scored):
        failures.append("raw/scored example-ID sets differ")

    observed_models: set[str] = set()
    observed_adapters: set[str] = set()
    observed_revisions: set[str] = set()

    for item_id, row in raw.items():
        observed_models.add(str(row.get("model")))
        observed_adapters.add(str(row.get("adapter")))
        if row.get("schema_version") != SCHEMA_VERSION:
            failures.append(
                f"{item_id}: schema={row.get('schema_version')!r}; expected {SCHEMA_VERSION!r}"
            )
        if row.get("correct") is not None:
            failures.append(f"{item_id}: correctness leaked into raw artifact")
        if row.get("reference") is not None:
            failures.append(f"{item_id}: reference leaked into raw artifact")
        meta = row.get("generation", {}).get("metadata", {})
        revision = meta.get("model_revision") or row.get("metadata", {}).get("model_revision")
        if not revision:
            failures.append(f"{item_id}: missing pinned model revision")
        else:
            observed_revisions.add(str(revision))
        if meta.get("trajectory_semantics") != "active_unresolved_positions_v4":
            failures.append(
                f"{item_id}: trajectory_semantics is not active_unresolved_positions_v4"
            )
        if not finite_float(meta.get("wall_time_seconds")) or float(meta["wall_time_seconds"]) <= 0:
            failures.append(f"{item_id}: missing/non-positive wall_time_seconds")
        if (
            not finite_float(meta.get("peak_gpu_memory_bytes"))
            or float(meta["peak_gpu_memory_bytes"]) <= 0
        ):
            failures.append(f"{item_id}: missing/non-positive peak_gpu_memory_bytes")

    if len(observed_models) != 1:
        failures.append(f"pilot mixes model IDs: {sorted(observed_models)}")
    if observed_adapters != {"llada"}:
        failures.append(f"pilot adapter must be llada; observed={sorted(observed_adapters)}")
    if len(observed_revisions) != 1:
        failures.append(
            f"pilot must use exactly one immutable model revision; "
            f"observed={sorted(observed_revisions)}"
        )

    if set(raw) == set(scored):
        for item_id in raw:
            if scored[item_id].get("correct") not in (True, False):
                failures.append(f"{item_id}: scored correctness is not boolean")
            if scored[item_id].get("reference") is not None:
                failures.append(f"{item_id}: reference stored in scored artifact")
            if raw[item_id].get("prompt_sha256") != scored[item_id].get("prompt_sha256"):
                failures.append(f"{item_id}: prompt hash changed after scoring")

    feature_matrix: list[list[float]] = []
    raw_risks: list[float] = []
    for item_id, row in raw.items():
        feats = row.get("generation", {}).get("features", {})
        missing = [name for name in PRIMARY_TRAJECTORY_FEATURES if name not in feats]
        if missing:
            failures.append(f"{item_id}: missing primary features={missing}")
            continue
        vector = [float(feats[name]) for name in PRIMARY_TRAJECTORY_FEATURES]
        if not np.all(np.isfinite(vector)):
            failures.append(f"{item_id}: non-finite primary feature")
            continue
        feature_matrix.append(vector)
        try:
            raw_risks.append(float(raw_trajectory_risk(feats)))
        except Exception as exc:
            failures.append(f"{item_id}: raw risk failed: {exc}")

    feature_stats: dict[str, dict[str, float | int]] = {}
    varying = 0
    if feature_matrix:
        x = np.asarray(feature_matrix, dtype=float)
        for j, name in enumerate(PRIMARY_TRAJECTORY_FEATURES):
            col = x[:, j]
            std = float(col.std(ddof=0))
            unique = int(len(np.unique(np.round(col, 12))))
            feature_stats[name] = {
                "mean": float(col.mean()),
                "std": std,
                "min": float(col.min()),
                "max": float(col.max()),
                "unique_rounded_12dp": unique,
            }
            if std > args.std_eps and unique > 1:
                varying += 1
    else:
        failures.append("no valid primary feature matrix")

    if varying < args.min_varying_primary_features:
        failures.append(
            f"only {varying}/{len(PRIMARY_TRAJECTORY_FEATURES)} primary features show "
            f"non-trivial variation; require >= {args.min_varying_primary_features}"
        )

    decision_std = feature_stats.get("decision_entropy_mean", {}).get("std", 0.0)
    if float(decision_std) <= args.std_eps:
        failures.append("decision_entropy_mean has no non-trivial variation")

    risk_std = float(np.std(raw_risks)) if raw_risks else 0.0
    if risk_std <= args.std_eps:
        failures.append("raw trajectory risk has no non-trivial variation")

    correctness = [
        bool(row["correct"])
        for row in scored_rows
        if row.get("correct") in (True, False)
    ]
    correct_count = int(sum(correctness))
    incorrect_count = int(len(correctness) - correct_count)
    if correct_count == 0 or incorrect_count == 0:
        failures.append(
            "pilot contains only one correctness class; calibration/error-detection "
            "diagnostics cannot be meaningfully exercised"
        )

    wall_times = [
        float(row["generation"]["metadata"]["wall_time_seconds"])
        for row in raw_rows
        if finite_float(row.get("generation", {}).get("metadata", {}).get("wall_time_seconds"))
    ]
    peak_mem = [
        float(row["generation"]["metadata"]["peak_gpu_memory_bytes"])
        for row in raw_rows
        if finite_float(row.get("generation", {}).get("metadata", {}).get("peak_gpu_memory_bytes"))
    ]
    step_counts = [len(row.get("generation", {}).get("steps", [])) for row in raw_rows]
    if any(count <= 0 for count in step_counts):
        failures.append("one or more pilot records has zero trajectory step summaries")

    commit = git_head()
    clean = git_clean()
    if not commit:
        failures.append("git commit could not be resolved")
    if clean is not True:
        failures.append("git worktree must be clean for the frozen pilot gate")

    payload = {
        "protocol": "trajuq_pilot_quality_gate_v2",
        "status": "PASS" if not failures else "FAIL",
        "git_commit": commit,
        "git_clean": clean,
        "schema_version": SCHEMA_VERSION,
        "model_ids": sorted(observed_models),
        "adapters": sorted(observed_adapters),
        "model_revisions": sorted(observed_revisions),
        "raw": args.raw,
        "raw_sha256": sha256_file(args.raw),
        "scored": args.scored,
        "scored_sha256": sha256_file(args.scored),
        "expected_records": args.expected_records,
        "raw_records": len(raw_rows),
        "scored_records": len(scored_rows),
        "correct": correct_count,
        "incorrect": incorrect_count,
        "varying_primary_features": varying,
        "primary_feature_count": len(PRIMARY_TRAJECTORY_FEATURES),
        "feature_stats": feature_stats,
        "raw_risk_std": risk_std,
        "wall_time_seconds": {
            "mean": float(np.mean(wall_times)) if wall_times else None,
            "p50": float(np.median(wall_times)) if wall_times else None,
            "p90": float(np.quantile(wall_times, 0.90)) if wall_times else None,
        },
        "peak_gpu_memory_bytes": {"max": int(max(peak_mem)) if peak_mem else None},
        "trajectory_step_count": {
            "min": int(min(step_counts)) if step_counts else None,
            "max": int(max(step_counts)) if step_counts else None,
            "unique": sorted(set(step_counts)),
        },
        "failures": failures,
        "warnings": warnings[:100],
        "manual_gate_remaining": "Inspect >=10 trajectory records manually even when automated status=PASS.",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
