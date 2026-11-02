#!/usr/bin/env python
"""Audit completed post-hoc systems diagnostics without granting claim authority."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ROOT / "results/systems"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_sidecar(directory: Path) -> list[dict]:
    declared = json.loads((directory / "SHA256SUMS.json").read_text(encoding="utf-8"))
    return [
        {
            "path": f"{directory.name}/{name}",
            "declared_sha256": digest,
            "actual_sha256": sha256(directory / name),
            "match": digest == sha256(directory / name),
        }
        for name, digest in sorted(declared.items())
    ]


def main() -> None:
    bench = SYSTEMS / "gpu_systems_bench_20261031"
    scaling = SYSTEMS / "context_decode_scaling_20261031"
    endurance = SYSTEMS / "endurance_20261031"
    preservation = SYSTEMS / "preservation_20261031"
    checks = check_sidecar(bench) + check_sidecar(scaling) + check_sidecar(endurance)
    mismatches = [row for row in checks if not row["match"]]
    expected_mismatch = [
        row for row in mismatches
        if row["path"] == "endurance_20261031/progress.json"
        and row["actual_sha256"] == "ebb85148ca5a606df92ea487df9cca1707e00c859ac04793c4ed5c4769264445"
    ]
    if len(mismatches) != 1 or len(expected_mismatch) != 1:
        raise SystemExit("unexpected systems SHA256SUMS mismatch")

    bench_rows = jsonl(bench / "systems_benchmark.jsonl")
    scaling_rows = jsonl(scaling / "context_decode_scaling.jsonl")
    endurance_rows = jsonl(endurance / "endurance_runs.jsonl")
    if len(bench_rows) != 150 or sum(bool(row["measured"]) for row in bench_rows) != 125:
        raise SystemExit("systems benchmark run-count drift")
    if len(scaling_rows) != 180 or sum(bool(row["measured"]) for row in scaling_rows) != 135:
        raise SystemExit("context/decode scaling run-count drift")
    if len(endurance_rows) != 375 or any(row.get("status") != "PASS" for row in endurance_rows):
        raise SystemExit("endurance completion drift")

    endurance_summary = json.loads((endurance / "endurance_summary.json").read_text(encoding="utf-8"))
    if endurance_summary["completed_runs"] != endurance_summary["expected_runs"] or not all(
        cell["deterministic_outputs"] and cell["successful_runs"] == 75
        for cell in endurance_summary["cells"]
    ):
        raise SystemExit("endurance determinism/completeness failure")

    archive = preservation / "archives/trajuq_systems_diagnostics_20261031.tar.gz"
    archive_sidecar = (Path(str(archive) + ".sha256").read_text(encoding="utf-8").split()[0])
    if sha256(archive) != archive_sidecar:
        raise SystemExit("systems preservation archive hash drift")
    watcher = preservation / "logs/TRAJUQ_WAIT_AND_PRESERVE_20261031.log"
    watcher_text = watcher.read_text(encoding="utf-8")
    if "TRAJUQ_SYSTEMS_DIAGNOSTICS_PRESERVATION=PASS" not in watcher_text:
        raise SystemExit("systems preservation success marker absent")

    preserved_hashes = {}
    for line in (preservation / "hashes/DIAGNOSTIC_FILES.sha256").read_text(encoding="utf-8").splitlines():
        digest, path = line.split(maxsplit=1)
        preserved_hashes[path.strip()] = digest
    required_preserved = {
        "/data/trajuq_gpu_systems_bench_20261031/systems_benchmark.jsonl": sha256(bench / "systems_benchmark.jsonl"),
        "/data/trajuq_gpu_context_decode_scaling_20261031/context_decode_scaling.jsonl": sha256(scaling / "context_decode_scaling.jsonl"),
        "/data/trajuq_gpu_endurance_20261031/endurance_runs.jsonl": sha256(endurance / "endurance_runs.jsonl"),
        "/data/trajuq_gpu_endurance_20261031/progress.json": sha256(endurance / "progress.json"),
    }
    for path, digest in required_preserved.items():
        if preserved_hashes.get(path) != digest:
            raise SystemExit(f"preservation ledger mismatch: {path}")

    output = {
        "schema": "trajuq_systems_diagnostics_audit_v1",
        "status": "PASS_WITH_PRESERVED_STALE_INTERNAL_PROGRESS_HASH",
        "classification": "SECONDARY_EXPLORATORY_SYSTEMS_DIAGNOSTIC",
        "scientific_claim_authority": False,
        "confirmatory_rq1_rq5": False,
        "uses_benchmark_prompts_labels_or_outcomes": False,
        "run_counts": {
            "systems_benchmark_total_including_warmup": len(bench_rows),
            "systems_benchmark_measured": 125,
            "context_decode_total_including_warmup": len(scaling_rows),
            "context_decode_measured": 135,
            "endurance_measured_pass": len(endurance_rows),
        },
        "oom_or_error_cells": 0,
        "endurance": {
            "expected_runs": 375,
            "completed_runs": 375,
            "all_cells_deterministic": True,
            "max_temperature_c": max(cell["max_temperature_c"] for cell in endurance_summary["cells"]),
            "max_absolute_first_last_quartile_drift_pct": max(
                abs(cell["throughput_drift_first_to_last_quartile_pct"])
                for cell in endurance_summary["cells"]
            ),
        },
        "hash_checks": checks,
        "preserved_discrepancy": {
            "path": "endurance_20261031/progress.json",
            "explanation": "The internal SHA256SUMS.json captured a pre-final progress file. The final COMPLETE progress file is bound correctly by the later preservation ledger and archive.",
            "internal_declared_sha256": mismatches[0]["declared_sha256"],
            "final_actual_and_preservation_sha256": mismatches[0]["actual_sha256"],
            "raw_artifacts_modified": False,
        },
        "preservation": {
            "watcher_success_marker": True,
            "archive_sha256": sha256(archive),
            "diagnostic_hash_ledger_sha256": sha256(preservation / "hashes/DIAGNOSTIC_FILES.sha256"),
        },
        "allowed_use": [
            "supplementary inference-cost characterization",
            "throughput, latency, VRAM, power, thermal, and reproducibility discussion",
        ],
        "forbidden_use": [
            "RQ1-RQ5 hypothesis authorization",
            "benchmark accuracy conclusions",
            "causal observability claims",
        ],
    }
    destination = SYSTEMS / "systems_diagnostics_audit.json"
    payload = (json.dumps(output, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with destination.open("xb") as handle:
        handle.write(payload)
    with Path(str(destination) + ".sha256").open("xb") as handle:
        handle.write((hashlib.sha256(payload).hexdigest() + "\n").encode())


if __name__ == "__main__":
    main()
