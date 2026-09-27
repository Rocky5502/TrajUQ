#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL = "trajuq_a100_rental_clock_v1"


def git_text(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def frozen_target_hours(path: Path) -> int:
    """Read target_wall_clock_hours without requiring PyYAML at bootstrap."""

    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"(?m)^\s*target_wall_clock_hours\s*:\s*([0-9]+)\s*$",
        text,
    )
    if not match:
        raise ValueError(
            f"{path}: cannot resolve integer target_wall_clock_hours"
        )
    hours = int(match.group(1))
    if hours <= 0:
        raise ValueError("target_wall_clock_hours must be positive")
    return hours


def iso(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Record the single immutable wall-clock deadline for a rented A100 "
            "session. Run immediately after connecting to the rental instance."
        )
    )
    parser.add_argument(
        "--primary-config",
        default="configs/experiments/primary.yaml",
    )
    parser.add_argument(
        "--output",
        default="results/a100_rental_clock.json",
    )
    args = parser.parse_args()

    config = Path(args.primary_config)
    if not config.is_file():
        raise SystemExit(f"Missing primary config: {config}")

    head = git_text("rev-parse", "HEAD")
    status = git_text("status", "--porcelain")
    if not head:
        raise SystemExit("Unable to resolve git HEAD.")
    if status is None:
        raise SystemExit("Unable to inspect git worktree.")
    if status:
        raise SystemExit(
            "Rental clock requires a clean checkout so its deadline is bound "
            "to the exact experiment commit."
        )

    try:
        target_hours = frozen_target_hours(config)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    out = Path(args.output)
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"Existing rental clock is invalid: {out}: {exc}"
            ) from exc
        if (
            existing.get("protocol") != PROTOCOL
            or existing.get("git_commit") != head
            or int(existing.get("target_wall_clock_hours", -1))
            != target_hours
        ):
            raise SystemExit(
                "Existing rental clock belongs to a different protocol, "
                "commit, or budget. Preserve it; do not silently restart time."
            )
        deadline = int(existing.get("deadline_epoch", 0))
        remaining = max(0, deadline - int(time.time()))
        print(
            json.dumps(
                {
                    "status": "PRESERVED",
                    "output": str(out),
                    "git_commit": head,
                    "target_wall_clock_hours": target_hours,
                    "remaining_hours": remaining / 3600.0,
                    "deadline_at_utc": existing.get("deadline_at_utc"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    start = int(time.time())
    deadline = start + target_hours * 3600
    payload = {
        "protocol": PROTOCOL,
        "status": "ACTIVE",
        "git_commit": head,
        "git_clean": True,
        "primary_config": str(config),
        "target_wall_clock_hours": target_hours,
        "start_epoch": start,
        "deadline_epoch": deadline,
        "started_at_utc": iso(start),
        "deadline_at_utc": iso(deadline),
        "definition": (
            "Single paid-rental wall-clock envelope. Setup, dependency install, "
            "downloads, parity, human gates, primary/RQ3/RQ2, and paper rendering "
            "all consume the same deadline."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(out)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
