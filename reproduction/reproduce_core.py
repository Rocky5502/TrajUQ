#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*parts: str) -> None:
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, *parts], cwd=ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce the authorized TrajUQ evidence layer.")
    parser.add_argument("--final", action="store_true", help="Require all registered RQ gates and run the final evidence pipeline.")
    args = parser.parse_args()
    run("scripts/validate_release_manifest.py")
    run("scripts/source_only_manuscript_audit.py")
    run("scripts/audit_paper.py", "--paper-dir", "paper")
    if args.final:
        run("scripts/paper_evidence_pipeline.py", "final")
        run("scripts/audit_paper.py", "--paper-dir", "paper", "--submission-ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

