#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import zipfile
from pathlib import Path


TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".jsonl", ".ps1", ".cff", ".tex", ".bib"}
PATTERNS = {
    "credential assignment": re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "Windows absolute path": re.compile(r"\b[A-Za-z]:\\"),
    "Unix home path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "public IPv4 address": re.compile(r"(?<![\d.])(?!(?:10|127|192\.168)\.)(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
    "development owner": re.compile(r"Rocky5502", re.IGNORECASE),
    "development username": re.compile(r"(?i)(?:Users\\rocky|/rocky/)"),
    "development Git remote": re.compile(r"(?i)(?:github\.com[:/]|ssh\.github\.com).*TrajUQ"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the final anonymous reviewer archive.")
    parser.add_argument("archive", nargs="?", default="dist/traj-uq-tmlr-anonymous.zip")
    parser.add_argument("--output", default="ANONYMIZED_ARTIFACT_AUDIT.json")
    args = parser.parse_args()
    archive = Path(args.archive).resolve()
    failures: list[str] = []
    warnings: list[str] = []
    names: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        names = sorted(zf.namelist())
        if len(names) != len(set(names)):
            failures.append("duplicate archive member")
        if "README.md" not in names or "MANIFEST.json" not in names:
            failures.append("missing reviewer README or MANIFEST")
        for name in names:
            if name.startswith("/") or ".." in Path(name).parts:
                failures.append(f"unsafe archive path: {name}")
                continue
            if Path(name).suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                text = zf.read(name).decode("utf-8")
            except UnicodeDecodeError:
                failures.append(f"non-UTF-8 text file: {name}")
                continue
            for label, pattern in PATTERNS.items():
                if name.startswith("tests/") and label in {"Windows absolute path", "Unix home path"}:
                    continue
                if pattern.search(text):
                    failures.append(f"{name}: {label}")
        with tempfile.TemporaryDirectory(prefix="trajuq-anon-audit-") as temporary:
            zf.extractall(temporary)
            if not (Path(temporary) / "README.md").is_file():
                failures.append("clean extraction missing README.md")
    license_present = "LICENSE" in names and "THIRD_PARTY.md" in names
    if not license_present:
        failures.append("license or third-party notice missing")
    report = {
        "schema": "trajuq_anonymized_artifact_audit_v1",
        "status": "PASS" if not failures else "FAIL",
        "archive": Path(args.archive).as_posix(),
        "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "file_count": len(names),
        "checks": [
            "anonymity", "secret", "personal_path", "ip_hostname",
            "absolute_path", "author_institution", "git_remote_owner",
            "license", "clean_extraction",
        ],
        "failures": failures,
        "warnings": warnings,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
