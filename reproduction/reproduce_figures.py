#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [sys.executable, "scripts/paper_evidence_pipeline.py", "final"],
        cwd=ROOT,
        env=env,
        check=True,
    )

