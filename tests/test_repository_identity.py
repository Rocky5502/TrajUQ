from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.repository_identity import (
    EXPECTED_REPOSITORY,
    verify_repository_identity,
)


def _repository(tmp_path: Path, remote: str) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".project-id").write_text(
        "TrajUQ\nRocky5502/TrajUQ\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "remote", "add", "origin", remote], cwd=root, check=True)
    return root


def test_repository_identity_accepts_expected_https_remote(tmp_path: Path) -> None:
    root = _repository(tmp_path, "https://github.com/Rocky5502/TrajUQ.git")
    assert verify_repository_identity(root) == EXPECTED_REPOSITORY


def test_repository_identity_accepts_expected_ssh_remote(tmp_path: Path) -> None:
    root = _repository(
        tmp_path, "ssh://git@ssh.github.com:443/Rocky5502/TrajUQ.git"
    )
    assert verify_repository_identity(root) == EXPECTED_REPOSITORY


def test_repository_identity_rejects_wrong_remote(tmp_path: Path) -> None:
    root = _repository(tmp_path, "https://github.com/Rocky5502/TraceCalib.git")
    with pytest.raises(RuntimeError, match="WRONG_REPOSITORY"):
        verify_repository_identity(root)


def test_repository_identity_rejects_missing_marker(tmp_path: Path) -> None:
    root = _repository(tmp_path, "https://github.com/Rocky5502/TrajUQ.git")
    (root / ".project-id").unlink()
    with pytest.raises(RuntimeError, match="WRONG_REPOSITORY"):
        verify_repository_identity(root)
