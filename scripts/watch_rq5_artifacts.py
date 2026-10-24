#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def run(command: list[str], log_path: Path) -> bool:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    entry = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return completed.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh RQ5 evidence and paper artifacts without touching API workers.")
    parser.add_argument("--refresh-seconds", type=int, default=300)
    parser.add_argument("--package-seconds", type=int, default=1800)
    parser.add_argument("--log", default="results/rq5/artifact_refresh.jsonl")
    args = parser.parse_args()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last_package = 0.0
    while True:
        cycle_started = time.monotonic()
        ledger_ok = run([sys.executable, "scripts/rq5_cost_ledger.py"], log_path)
        registry_ok = run([sys.executable, "scripts/build_rq5_validated_registry.py"], log_path)
        if ledger_ok and registry_ok and cycle_started - last_package >= args.package_seconds:
            if run([sys.executable, "scripts/build_overleaf_package.py"], log_path):
                last_package = cycle_started
        elapsed = time.monotonic() - cycle_started
        time.sleep(max(1.0, args.refresh_seconds - elapsed))


if __name__ == "__main__":
    main()
