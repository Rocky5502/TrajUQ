#!/usr/bin/env python
"""Build a deterministic TrajUQ code/reproducibility archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

try:
    from scripts.repository_identity import verify_repository_identity
except ModuleNotFoundError:
    from repository_identity import verify_repository_identity

PROTOCOL = "trajuq_code_release_v2"
EXCLUDED_TOP_LEVEL = {
    "dist",
    "tmp",
    "output",
    "paper",
    "manuscript",
    "overleaf",
    ".ruff_cache",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def git_paths(root: Path) -> list[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return sorted(item.decode("utf-8") for item in output.split(b"\0") if item)


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_files(root: Path, output_dir: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    resolved_output = output_dir.resolve()
    for name in git_paths(root):
        posix = PurePosixPath(name)
        if not posix.parts:
            continue
        top = posix.parts[0]
        if top in EXCLUDED_TOP_LEVEL or top.startswith(".pytest"):
            continue
        path = root.joinpath(*posix.parts)
        resolved = path.resolve()
        if resolved == resolved_output or resolved_output in resolved.parents:
            continue
        if path.is_file():
            files[name] = path.read_bytes()
    if not files:
        raise ValueError("release archive would be empty")
    return files


def zip_bytes(files: dict[str, bytes]) -> bytes:
    with tempfile.SpooledTemporaryFile() as stream:
        with zipfile.ZipFile(
            stream,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, data in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        stream.seek(0)
        return stream.read()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def build_release(*, root: Path, output_dir: Path, release_date: str) -> dict[str, object]:
    if len(release_date) != 8 or not release_date.isdigit():
        raise ValueError("release date must be YYYYMMDD")
    root = root.resolve()
    output_dir = output_dir.resolve()
    verify_repository_identity(root)

    commit = git_commit(root)
    files = collect_files(root, output_dir)
    internal = {
        "files": [
            {"bytes": len(data), "path": name, "sha256": sha256_bytes(data)}
            for name, data in sorted(files.items())
        ],
        "kind": "CODE",
        "protocol": PROTOCOL,
        "source_git_commit": commit,
    }
    files = {**files, "SHA256_MANIFEST.json": canonical_json(internal)}
    archive_name = f"TrajUQ_CODE_{release_date}.zip"
    archive_data = zip_bytes(files)
    archive_path = output_dir / archive_name
    atomic_write(archive_path, archive_data)

    payload = {
        "archive": {
            "bytes": len(archive_data),
            "file_count": len(files),
            "kind": "CODE",
            "path": archive_name,
            "sha256": sha256_bytes(archive_data),
        },
        "protocol": PROTOCOL,
        "release_date": release_date,
        "source_git_commit": commit,
    }
    manifest_path = output_dir / f"TrajUQ_BUILD_{release_date}.json"
    atomic_write(manifest_path, canonical_json(payload))
    return {**payload, "manifest": manifest_path.name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="dist")
    parser.add_argument("--date", default=date.today().strftime("%Y%m%d"))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    try:
        payload = build_release(root=root, output_dir=output_dir, release_date=args.date)
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError) as exc:
        print(canonical_json({"reason": str(exc), "status": "FAIL"}).decode(), end="")
        raise SystemExit(1) from exc
    print(canonical_json({**payload, "status": "PASS"}).decode(), end="")


if __name__ == "__main__":
    main()
