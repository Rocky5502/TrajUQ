from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Mapping


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def analysis_provenance(
    sources: Mapping[str, str | Path],
) -> dict:
    commit = git_text("rev-parse", "HEAD")
    status = git_text("status", "--porcelain")
    return {
        "protocol": "trajuq_analysis_source_binding_v1",
        "git_commit": commit,
        "git_clean": None if status is None else status == "",
        "sources": {
            str(name): {
                "path": str(Path(path)),
                "sha256": sha256_file(path),
            }
            for name, path in sources.items()
        },
    }


def compact_row_binding(provenance: dict) -> dict:
    sources = provenance.get("sources", {})
    return {
        "protocol": provenance.get("protocol"),
        "git_commit": provenance.get("git_commit"),
        "git_clean": provenance.get("git_clean"),
        "source_sha256": {
            str(name): str(spec.get("sha256", ""))
            for name, spec in sources.items()
        },
    }


def binding_matches(
    provenance: dict,
    *,
    expected_commit: str,
    expected_sources: Mapping[str, tuple[str | Path, str]],
) -> bool:
    if provenance.get("protocol") != "trajuq_analysis_source_binding_v1":
        return False
    if provenance.get("git_commit") != expected_commit:
        return False
    if provenance.get("git_clean") is not True:
        return False
    observed = provenance.get("sources", {})
    if set(observed) != set(expected_sources):
        return False
    for name, (path, digest) in expected_sources.items():
        spec = observed.get(name, {})
        if str(Path(str(spec.get("path", "")))) != str(Path(path)):
            return False
        if str(spec.get("sha256", "")) != str(digest):
            return False
    return True


def compact_binding_matches(
    binding: dict,
    *,
    expected_commit: str,
    expected_sha256: Mapping[str, str],
) -> bool:
    return (
        binding.get("protocol") == "trajuq_analysis_source_binding_v1"
        and binding.get("git_commit") == expected_commit
        and binding.get("git_clean") is True
        and binding.get("source_sha256")
        == {str(name): str(value) for name, value in expected_sha256.items()}
    )
