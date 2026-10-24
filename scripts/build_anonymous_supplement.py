#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path


INCLUDE_ROOTS = [
    "src",
    "scripts",
    "configs",
    "tests",
    "data/manifests",
    "data/samples",
    "docs",
]

TOP_LEVEL_FILES = [
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
    "THIRD_PARTY.md",
    "EXPERIMENT_FREEZE_CHECKLIST.md",
]

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".jsonl",
    ".ps1",
    ".cff",
    ".tex",
    ".bib",
}

EXCLUDED_PATH_PREFIXES = {
    "paper/",
    "results/",
    "runs/",
    ".github/",
    ".git/",
    "dist/",
}

EXCLUDED_EXACT = {
    "README.md",
    "CITATION.cff",
    "docs/NAACL2027_SUBMISSION.md",
    # Internal server handoff contains the development repository identity and
    # is not part of the anonymous scientific software supplement.
    "docs/A100_EXECUTION.md",
    "docs/CODEX_A100_72H.md",
    # The builder itself contains the development-identity denylist and is not
    # needed by reviewers to run the submitted software.
    "scripts/build_anonymous_supplement.py",
    # Repository/release safety tooling intentionally binds the development
    # repository identity. It is operational infrastructure, not scientific
    # reviewer code, and must never enter the anonymous supplement.
    "scripts/package_release.py",
    "scripts/repository_identity.py",
    "scripts/safe_push.py",
    "tests/test_package_release.py",
    "tests/test_repository_identity.py",
}

PATH_PATTERN_LABELS = {
    "Windows absolute path",
    "Unix home path",
}

BANNED_PATTERNS = {
    "development repository owner": re.compile(r"Rocky5502", re.IGNORECASE),
    "development repository URL": re.compile(
        r"https?://(?:www\.)?github\.com/Rocky5502/TrajUQ",
        re.IGNORECASE,
    ),
    "Windows absolute path": re.compile(r"\b[A-Za-z]:\\"),
    "Unix home path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "Dropbox tracking link": re.compile(r"https?://(?:www\.)?dropbox\.", re.IGNORECASE),
    "Google Drive tracking link": re.compile(
        r"https?://drive\.google\.com/",
        re.IGNORECASE,
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def should_include(path: Path) -> bool:
    rel = path.as_posix()
    if rel in EXCLUDED_EXACT:
        return False
    if any(rel.startswith(prefix) for prefix in EXCLUDED_PATH_PREFIXES):
        return False
    if rel in TOP_LEVEL_FILES:
        return True
    return any(rel == root or rel.startswith(root + "/") for root in INCLUDE_ROOTS)


def collect_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if should_include(rel):
            files.append(rel)
    return sorted(files, key=lambda p: p.as_posix())


def scan_text(rel: Path, data: bytes) -> list[str]:
    if rel.suffix.lower() not in TEXT_SUFFIXES:
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [f"{rel}: text-like file is not valid UTF-8"]

    failures: list[str] = []
    for label, pattern in BANNED_PATTERNS.items():
        # Unit tests intentionally contain synthetic absolute-path fixtures.
        # Keep identity/repository scans active for tests, but do not treat
        # those path fixtures as leaks in the anonymous software bundle.
        if rel.parts and rel.parts[0] == "tests" and label in PATH_PATTERN_LABELS:
            continue
        if pattern.search(text):
            failures.append(f"{rel}: contains {label}")
    return failures


def zip_write_bytes(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic anonymized ARR software supplement."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="dist/trajuq_anonymous_software.zip",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=200 * 1024 * 1024,
        help="Maximum archive size; ARR software uploads are capped at 200 MB.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if root not in output.parents:
        raise SystemExit("Output must be inside repository root.")

    anon_readme = root / "supplement" / "README.md"
    if not anon_readme.exists():
        raise SystemExit("Missing supplement/README.md")

    files = collect_files(root)
    if not files:
        raise SystemExit("No supplement files selected.")

    failures: list[str] = []
    entries: list[dict[str, str | int]] = []

    supplement_readme_data = anon_readme.read_bytes()
    failures.extend(scan_text(Path("README.md"), supplement_readme_data))

    for rel in files:
        data = (root / rel).read_bytes()
        failures.extend(scan_text(rel, data))
        entries.append(
            {
                "path": rel.as_posix(),
                "sha256": sha256_bytes(data),
                "bytes": len(data),
            }
        )

    if failures:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "failures": failures,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(1)

    manifest = {
        "format": "trajuq-anonymous-software-v1",
        "note": (
            "Generated deterministically from the development repository. "
            "No raw benchmark caches, model weights, run outputs, paper source, "
            "author metadata, or development-repository link are included."
        ),
        "files": [
            {
                "path": "README.md",
                "sha256": sha256_bytes(supplement_readme_data),
                "bytes": len(supplement_readme_data),
            },
            *entries,
        ],
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with zipfile.ZipFile(output, "w") as zf:
        zip_write_bytes(zf, "README.md", supplement_readme_data)
        for rel in files:
            zip_write_bytes(zf, rel.as_posix(), (root / rel).read_bytes())
        zip_write_bytes(zf, "MANIFEST.json", manifest_bytes)

    archive_size = output.stat().st_size
    if archive_size > args.max_bytes:
        output.unlink()
        raise SystemExit(
            f"Anonymous software archive exceeds size limit: "
            f"{archive_size} > {args.max_bytes} bytes"
        )

    archive_sha = sha256_bytes(output.read_bytes())
    payload = {
        "status": "PASS",
        "output": str(output.relative_to(root)),
        "archive_sha256": archive_sha,
        "file_count": len(entries) + 2,
        "archive_bytes": archive_size,
        "max_bytes": args.max_bytes,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
