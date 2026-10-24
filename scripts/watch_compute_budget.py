#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def run(command: list[str], log: Path) -> bool:
    p = subprocess.run(command, text=True, capture_output=True, check=False)
    with log.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "command": command,
                    "returncode": p.returncode,
                    "stdout": p.stdout[-2000:],
                    "stderr": p.stderr[-2000:],
                },
                sort_keys=True,
            )
            + "\n"
        )
    return p.returncode == 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--credentials", required=True)
    p.add_argument("--refresh-seconds", type=int, default=600)
    p.add_argument("--system-python", default=sys.executable)
    args = p.parse_args()
    log = Path("results/compute_budget/refresh.jsonl")
    log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        started = time.monotonic()
        ok = run(
            [
                args.system_python,
                "scripts/capture_compute_environment.py",
                "--credentials",
                args.credentials,
            ],
            log,
        )
        if ok:
            run([sys.executable, "scripts/build_compute_budget.py"], log)
            run([sys.executable, "scripts/build_overleaf_package.py"], log)
        time.sleep(max(1, args.refresh_seconds - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
