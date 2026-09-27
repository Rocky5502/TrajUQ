#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from trajuq.io import read_jsonl, write_json_immutable


def index(path: str) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    out: dict[str, dict] = {}
    for row in rows:
        item_id = str(row.get("example_id", ""))
        if not item_id:
            raise SystemExit(f"{path}: missing example_id")
        if item_id in out:
            raise SystemExit(f"{path}: duplicate example_id={item_id}")
        out[item_id] = row
    return out


def sha256_file(path: str | Path) -> str:
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


def run_signature(rows: dict[str, dict]) -> dict[str, object]:
    if not rows:
        raise ValueError("overhead source run contains zero records")
    fields: dict[str, set[object]] = {
        "dataset": set(),
        "model": set(),
        "adapter": set(),
        "seed": set(),
        "model_revision": set(),
        "source_git_commit": set(),
        "source_git_clean": set(),
    }
    for row in rows.values():
        fields["dataset"].add(row.get("dataset"))
        fields["model"].add(row.get("model"))
        fields["adapter"].add(row.get("adapter"))
        fields["seed"].add(row.get("seed"))
        metadata = row.get("generation", {}).get("metadata", {})
        fields["model_revision"].add(
            metadata.get("model_revision") if isinstance(metadata, dict) else None
        )
        record_metadata = row.get("metadata", {})
        fields["source_git_commit"].add(
            record_metadata.get("source_git_commit")
            if isinstance(record_metadata, dict)
            else None
        )
        fields["source_git_clean"].add(
            record_metadata.get("source_git_clean")
            if isinstance(record_metadata, dict)
            else None
        )
    signature: dict[str, object] = {}
    for key, values in fields.items():
        if None in values or len(values) != 1:
            raise ValueError(
                f"overhead source has non-uniform/missing {key}: "
                f"{sorted(str(value) for value in values)}"
            )
        signature[key] = next(iter(values))
    return signature


def metric(row: dict, name: str):
    value = row.get("generation", {}).get("metadata", {}).get(name)
    if value is None:
        return None
    return float(value)


def bootstrap_mean_delta(
    values: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
):
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        means.append(float(values[idx].mean()))
    low, high = np.quantile(np.asarray(means), [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired instrumentation-overhead audit for native vs TrajUQ runs."
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--instrumented", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    reference = index(args.reference)
    instrumented = index(args.instrumented)
    if set(reference) != set(instrumented):
        raise SystemExit("Reference/instrumented ID sets differ")

    try:
        reference_signature = run_signature(reference)
        instrumented_signature = run_signature(instrumented)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if reference_signature != instrumented_signature:
        raise SystemExit(
            "Reference/instrumented run provenance differs: "
            f"{reference_signature} != {instrumented_signature}"
        )

    commit = git_head()
    clean = git_clean()
    if not commit:
        raise SystemExit("Instrumentation-overhead audit requires a resolvable git commit.")
    if clean is not True:
        raise SystemExit(
            "Instrumentation-overhead audit requires the frozen clean checkout."
        )
    if reference_signature.get("source_git_commit") != commit:
        raise SystemExit(
            "Instrumentation-overhead sources were generated from a different git commit."
        )
    if reference_signature.get("source_git_clean") is not True:
        raise SystemExit(
            "Instrumentation-overhead sources were not generated from a clean checkout."
        )

    ids = sorted(reference)
    native_time = []
    traj_time = []
    native_mem = []
    traj_mem = []

    for item_id in ids:
        nt = metric(reference[item_id], "wall_time_seconds")
        tt = metric(instrumented[item_id], "wall_time_seconds")
        if nt is None or tt is None or nt <= 0 or tt <= 0:
            raise SystemExit(f"{item_id}: missing/non-positive timing metadata")
        native_time.append(nt)
        traj_time.append(tt)

        nm = metric(reference[item_id], "peak_gpu_memory_bytes")
        tm = metric(instrumented[item_id], "peak_gpu_memory_bytes")
        if nm is not None and tm is not None:
            native_mem.append(nm)
            traj_mem.append(tm)

    native_time_arr = np.asarray(native_time, dtype=float)
    traj_time_arr = np.asarray(traj_time, dtype=float)
    relative_time_delta = traj_time_arr / native_time_arr - 1.0
    time_point, time_low, time_high = bootstrap_mean_delta(
        relative_time_delta,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    payload = {
        "protocol": "trajuq_instrumentation_overhead_v2",
        "git_commit": commit,
        "git_clean": clean,
        "reference": str(Path(args.reference)),
        "reference_sha256": sha256_file(args.reference),
        "instrumented": str(Path(args.instrumented)),
        "instrumented_sha256": sha256_file(args.instrumented),
        "run_provenance": reference_signature,
        "n": len(ids),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "time_seconds": {
            "native_mean": float(native_time_arr.mean()),
            "instrumented_mean": float(traj_time_arr.mean()),
            "paired_relative_delta_mean": time_point,
            "paired_relative_delta_ci_low": time_low,
            "paired_relative_delta_ci_high": time_high,
            "median_instrumented_over_native_ratio": float(
                np.median(traj_time_arr / native_time_arr)
            ),
        },
        "memory_bytes": None,
    }

    if native_mem and len(native_mem) == len(ids):
        native_mem_arr = np.asarray(native_mem, dtype=float)
        traj_mem_arr = np.asarray(traj_mem, dtype=float)
        payload["memory_bytes"] = {
            "native_max": int(native_mem_arr.max()),
            "instrumented_max": int(traj_mem_arr.max()),
            "max_delta": int(traj_mem_arr.max() - native_mem_arr.max()),
            "paired_mean_delta": float((traj_mem_arr - native_mem_arr).mean()),
        }

    out = Path(args.output)
    write_json_immutable(out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
