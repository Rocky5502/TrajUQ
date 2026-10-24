#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception:
        return None


def run_step(name: str, argv: list[str], logs_dir: Path) -> dict:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    status = "PASS" if completed.returncode == 0 else "FAIL"
    print(f"[{status}] {name}: {' '.join(argv)}")
    if completed.stdout:
        for line in completed.stdout.splitlines()[-12:]:
            print(line)
    return {
        "name": name,
        "argv": argv,
        "status": status,
        "returncode": int(completed.returncode),
        "duration_seconds": float(elapsed),
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete TrajUQ CPU verification suite."
    )
    parser.add_argument("--output", default="results/cpu_verification.json")
    parser.add_argument("--logs-dir", default="results/cpu_verification_logs")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    logs_path = Path(args.logs_dir)
    existing_logs = [p for p in logs_path.iterdir() if p.is_file()] if logs_path.is_dir() else []
    if (output_path.exists() or existing_logs) and not args.overwrite:
        existing = ([output_path] if output_path.exists() else []) + existing_logs
        raise SystemExit(
            "Refusing to overwrite CPU verification evidence: "
            + ", ".join(str(path) for path in existing[:12])
        )

    commit = git_text("rev-parse", "HEAD")
    dirty_text = git_text("status", "--porcelain")
    dirty = bool(dirty_text)
    failures: list[str] = []
    if not commit:
        failures.append("git commit could not be resolved")
    if dirty and not args.allow_dirty:
        failures.append("git worktree is dirty")

    py = sys.executable
    math500_audit = [
        py,
        "scripts/audit_math500_scorer.py",
        "--output",
        "results/math500_scorer_audit.json",
    ]
    if args.overwrite:
        math500_audit.append("--overwrite")

    steps = [
        ("security_scan", [py, "scripts/security_scan.py", "--repo", "."]),
        ("compileall", [py, "-m", "compileall", "-q", "src", "scripts", "tests"]),
        ("ruff", [py, "-m", "ruff", "check", "src", "tests", "scripts"]),
        ("pytest", [py, "-m", "pytest", "-q", "--basetemp", "results/pytest_tmp"]),
        ("metric_smoke", [py, "scripts/smoke_metrics.py"]),
        (
            "end_to_end_smoke",
            [py, "scripts/smoke_end_to_end.py", "--output-dir", "results/cpu_smoke_e2e"],
        ),
        ("math500_scorer_audit", math500_audit),
    ]

    records: list[dict] = []
    if not failures or args.allow_dirty:
        for name, argv in steps:
            record = run_step(name, argv, logs_path)
            records.append(record)
            if record["status"] != "PASS":
                failures.append(f"{name} failed with exit code {record['returncode']}")
                if not args.continue_on_failure:
                    break

    payload = {
        "protocol": "trajuq_cpu_verification_v2",
        "status": "PASS" if not failures else "FAIL",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_clean": not dirty,
        "git_status_porcelain": dirty_text or "",
        "python": sys.version,
        "platform": platform.platform(),
        "steps": records,
        "expected_steps": [name for name, _ in steps],
        "completed_steps": [record["name"] for record in records],
        "failures": failures,
        "note": (
            "This verifies CPU-side repository integrity only. Model-specific "
            "GPU parity and scientific evidence gates remain separate."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
