#!/usr/bin/env python
"""Build deterministic, disjoint CODE and OVERLEAF release archives.

The CODE archive contains repository code/research material but never the
``paper/`` tree.  The OVERLEAF archive is resolved from ``paper/main.tex`` and
contains only files needed to compile the manuscript.  ZIP metadata is fixed,
and every archive has a SHA-256-bound manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.request
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

try:
    from scripts.repository_identity import verify_repository_identity
except ModuleNotFoundError:  # direct ``python scripts/package_release.py``
    from repository_identity import verify_repository_identity


PROTOCOL = "trajuq_release_package_v1"
TMLR_STYLE_COMMIT = "7bf90efe3a0debbba703c05c43f3ff7e4d4a2992"
TMLR_STYLE_BASE = (
    "https://raw.githubusercontent.com/JmlrOrg/tmlr-style-file/"
    + TMLR_STYLE_COMMIT
)
STYLE_FILES = ("tmlr.sty", "tmlr.bst", "fancyhdr.sty")
OVERLEAF_TEXT_SUFFIXES = {".tex", ".bib", ".sty", ".bst"}
OVERLEAF_ASSET_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".eps"}
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}")
BIB_RE = re.compile(r"\\bibliography\s*\{([^}]+)\}")
BANNED_OVERLEAF_PATH_PARTS = {
    "build-rq1",
    "__pycache__",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def manifest_entries(files: dict[str, bytes]) -> list[dict[str, object]]:
    return [
        {"bytes": len(data), "path": name, "sha256": sha256_bytes(data)}
        for name, data in sorted(files.items())
    ]


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
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
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


def git_paths(root: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
    )
    return sorted(item.decode("utf-8") for item in output.split(b"\0") if item)


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_code_files(root: Path, *, excluded_paths: tuple[Path, ...] = ()) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    excluded = tuple(path.resolve() for path in excluded_paths)
    for name in git_paths(root):
        posix = PurePosixPath(name)
        if not posix.parts:
            continue
        top = posix.parts[0]
        if (
            top in {"paper", "dist", "tmp", "$stage", ".ruff_cache"}
            or top.startswith(".pytest")
            or top.startswith("dist-")
        ):
            continue
        path = root.joinpath(*posix.parts)
        resolved = path.resolve()
        if any(resolved == item or item in resolved.parents for item in excluded):
            continue
        if path.is_file():
            files[name] = path.read_bytes()
    if not files:
        raise ValueError("CODE archive would be empty")
    return files


def resolve_paper_path(paper: Path, value: str, suffixes: tuple[str, ...]) -> Path | None:
    raw = value.strip().replace("\\", "/")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe manuscript dependency path: {value}")
    candidates = [paper.joinpath(*relative.parts)]
    if not relative.suffix:
        candidates.extend(paper.joinpath(*relative.parts).with_suffix(s) for s in suffixes)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def read_style(name: str, style_dir: Path | None) -> bytes:
    if style_dir is not None:
        path = style_dir / name
        if not path.is_file():
            raise ValueError(f"missing pinned style file: {path}")
        return path.read_bytes()
    request = urllib.request.Request(
        f"{TMLR_STYLE_BASE}/{name}",
        headers={"User-Agent": "TrajUQ-Release-Packager/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()
    if not data:
        raise ValueError(f"downloaded style file is empty: {name}")
    return data


def collect_overleaf_files(root: Path, style_dir: Path | None) -> dict[str, bytes]:
    paper = (root / "paper").resolve()
    main = paper / "main.tex"
    if not main.is_file():
        raise ValueError(f"missing manuscript entry point: {main}")

    files: dict[str, bytes] = {}
    pending = [main]
    visited: set[Path] = set()
    missing: list[str] = []
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        relative = source.relative_to(paper).as_posix()
        if any(part in BANNED_OVERLEAF_PATH_PARTS for part in source.parts):
            raise ValueError(f"non-manuscript build artifact selected: {relative}")
        data = source.read_bytes()
        files[relative] = data
        if source.suffix.lower() != ".tex":
            continue
        text = data.decode("utf-8")
        for value in INPUT_RE.findall(text):
            dependency = resolve_paper_path(paper, value, (".tex",))
            if dependency is None:
                # Optional \IfFileExists inputs are permitted to be absent.
                if f"\\IfFileExists{{{value}.tex}}" not in text and f"\\IfFileExists{{{value}}}" not in text:
                    missing.append(value)
            else:
                pending.append(dependency.resolve())
        for group in BIB_RE.findall(text):
            for value in group.split(","):
                dependency = resolve_paper_path(paper, value, (".bib",))
                if dependency is None:
                    missing.append(value)
                else:
                    pending.append(dependency.resolve())
        for value in GRAPHICS_RE.findall(text):
            dependency = resolve_paper_path(
                paper, value, tuple(sorted(OVERLEAF_ASSET_SUFFIXES))
            )
            if dependency is None:
                if f"\\IfFileExists{{{value}}}" not in text:
                    missing.append(value)
            else:
                pending.append(dependency.resolve())

    if missing:
        raise ValueError("missing manuscript dependencies: " + ", ".join(sorted(set(missing))))
    for name in STYLE_FILES:
        files[name] = read_style(name, style_dir)
    forbidden = [
        name
        for name in files
        if PurePosixPath(name).suffix.lower()
        not in OVERLEAF_TEXT_SUFFIXES | OVERLEAF_ASSET_SUFFIXES
    ]
    if forbidden:
        raise ValueError("non-manuscript files selected: " + ", ".join(forbidden))
    return files


def add_internal_manifest(
    files: dict[str, bytes], *, kind: str, source_commit: str | None
) -> dict[str, bytes]:
    payload = {
        "files": manifest_entries(files),
        "kind": kind,
        "protocol": PROTOCOL,
        "source_git_commit": source_commit,
        "tmlr_style_commit": TMLR_STYLE_COMMIT if kind == "OVERLEAF" else None,
    }
    return {**files, "SHA256_MANIFEST.json": canonical_json(payload)}


def build_release(
    *, root: Path, output_dir: Path, release_date: str, style_dir: Path | None = None
) -> dict[str, object]:
    if not re.fullmatch(r"\d{8}", release_date):
        raise ValueError("release date must be YYYYMMDD")
    root = root.resolve()
    verify_repository_identity(root)
    output_dir = output_dir.resolve()
    commit = git_commit(root)
    code_files = add_internal_manifest(
        collect_code_files(root, excluded_paths=(output_dir,)),
        kind="CODE",
        source_commit=commit,
    )
    overleaf_files = add_internal_manifest(
        collect_overleaf_files(root, style_dir),
        kind="OVERLEAF",
        source_commit=commit,
    )
    if set(code_files) & set(overleaf_files) - {"SHA256_MANIFEST.json"}:
        raise ValueError("CODE and OVERLEAF archive paths overlap")

    archives: list[dict[str, object]] = []
    for kind, files in (("CODE", code_files), ("OVERLEAF", overleaf_files)):
        name = f"TrajUQ_TMLR_{kind}_{release_date}.zip"
        data = zip_bytes(files)
        path = output_dir / name
        atomic_write(path, data)
        archives.append(
            {
                "bytes": len(data),
                "file_count": len(files),
                "kind": kind,
                "path": name,
                "sha256": sha256_bytes(data),
            }
        )
    manifest = {
        "archives": archives,
        "protocol": PROTOCOL,
        "release_date": release_date,
        "source_git_commit": commit,
    }
    manifest_path = output_dir / f"TrajUQ_TMLR_BUILD_{release_date}.json"
    atomic_write(manifest_path, canonical_json(manifest))
    return {**manifest, "manifest": manifest_path.name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="dist")
    parser.add_argument("--date", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--tmlr-style-dir")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    style_dir = Path(args.tmlr_style_dir).resolve() if args.tmlr_style_dir else None
    try:
        payload = build_release(
            root=root,
            output_dir=output_dir,
            release_date=args.date,
            style_dir=style_dir,
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError) as exc:
        print(canonical_json({"reason": str(exc), "status": "FAIL"}).decode(), end="")
        raise SystemExit(1) from exc
    print(canonical_json({**payload, "status": "PASS"}).decode(), end="")


if __name__ == "__main__":
    main()
