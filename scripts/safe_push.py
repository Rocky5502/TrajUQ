#!/usr/bin/env python
"""Push TrajUQ only after the repository identity guard passes."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

try:
    from scripts.repository_identity import verify_repository_identity
except ModuleNotFoundError:  # direct ``python scripts/safe_push.py``
    from repository_identity import verify_repository_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("git_push_arg", nargs="*", help="arguments forwarded to git push")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_repository_identity(root)
    subprocess.run(["git", "push", *args.git_push_arg], cwd=root, check=True)


if __name__ == "__main__":
    main()
