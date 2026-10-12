#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except (OSError, SystemError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce the RQ5 operational budget authorization.")
    parser.add_argument("--interval-seconds", type=int, default=30)
    parser.add_argument("--pid-file", default="results/rq5/api/run.pid")
    parser.add_argument("--ledger", default="results/rq5/api/cost_ledger.json")
    parser.add_argument("--events", default="results/rq5/api/budget_watch_events.jsonl")
    args = parser.parse_args()
    while True:
        subprocess.run([sys.executable, "scripts/rq5_cost_ledger.py"], check=True, stdout=subprocess.DEVNULL)
        ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
        pid_path = Path(args.pid_file)
        pid = int(pid_path.read_text().strip()) if pid_path.exists() else -1
        if ledger["cost_watch_mode"] == "HARD_STOP" or not ledger["paid_execution_allowed"]:
            if pid > 0 and process_alive(pid):
                os.kill(pid, signal.SIGTERM)
            event = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "event": "PAID_EXECUTION_STOPPED_BY_BUDGET_WATCH",
                "pid": pid,
                "cumulative_cost_rmb": ledger["cumulative_cost_rmb"],
                "projected_completion_cost_rmb": ledger["projected_completion_cost_rmb"],
                "mode": ledger["cost_watch_mode"],
            }
            events = Path(args.events)
            events.parent.mkdir(parents=True, exist_ok=True)
            with events.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            return
        if pid > 0 and not process_alive(pid):
            return
        time.sleep(max(5, args.interval_seconds))


if __name__ == "__main__":
    main()
