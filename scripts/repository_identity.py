#!/usr/bin/env python
"""Fail-closed repository identity checks for TrajUQ release operations."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


EXPECTED_PROJECT = "TrajUQ"
EXPECTED_REPOSITORY = "Rocky5502/TrajUQ"
PROJECT_ID_FILE = ".project-id"


def _repository_slug(remote_url: str) -> str | None:
    """Return an owner/repository slug without echoing credentials or hosts."""
    clean = remote_url.strip().rstrip("/")
    match = re.search(
        r"(?:github\.com[/:]|ssh\.github\.com(?::\d+)?/)([^/\s:]+)/([^/\s]+?)(?:\.git)?$",
        clean,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}"


def verify_repository_identity(root: Path, remote: str = "origin") -> str:
    """Verify the project marker and exact GitHub destination or fail closed."""
    root = root.resolve()
    marker = root / PROJECT_ID_FILE
    try:
        marker_lines = [
            line.strip()
            for line in marker.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        raise RuntimeError("WRONG_REPOSITORY: missing project identity marker") from exc
    if marker_lines != [EXPECTED_PROJECT, EXPECTED_REPOSITORY]:
        raise RuntimeError("WRONG_REPOSITORY: invalid project identity marker")

    try:
        remote_url = subprocess.check_output(
            ["git", "remote", "get-url", remote],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("WRONG_REPOSITORY: configured remote is unavailable") from exc
    slug = _repository_slug(remote_url)
    if slug is None or slug.lower() != EXPECTED_REPOSITORY.lower():
        raise RuntimeError("WRONG_REPOSITORY: expected Rocky5502/TrajUQ")
    return EXPECTED_REPOSITORY
