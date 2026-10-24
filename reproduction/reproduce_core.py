#!/usr/bin/env python
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*parts: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, *parts], cwd=ROOT, env=env, check=True)


def main() -> int:
    run("scripts/security_scan.py", "--repo", ".")
    run("scripts/validate_release_manifest.py")
    print("TRAJUQ_CORE_REPRODUCTION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
