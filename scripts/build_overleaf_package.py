#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path


ACL_STYLE_COMMIT = "d5adc823ff0f80f98c80405ca0ab66c68e684409"
ACL_STYLE_BASE = (
    "https://raw.githubusercontent.com/acl-org/acl-style-files/"
    + ACL_STYLE_COMMIT
)
STYLE_FILES = ("acl.sty", "acl_natbib.bst")
PAPER_FILES = (
    "main.tex",
    "appendix.tex",
    "references.bib",
    "generated_results.tex",
)
BANNED_PATTERNS = {
    "development repository owner": re.compile(
        "Rocky" + "5502",
        re.IGNORECASE,
    ),
    "public GitHub URL": re.compile(
        r"https?://(?:www\.)?github\.com/",
        re.IGNORECASE,
    ),
    "Windows absolute path": re.compile(r"\b[A-Za-z]:\\"),
    "Unix home path": re.compile(r"/home/[^/\s]+/"),
    "Dropbox tracking link": re.compile(
        r"https?://(?:www\.)?dropbox\.",
        re.IGNORECASE,
    ),
    "Google Drive tracking link": re.compile(
        r"https?://drive\.google\.com/",
        re.IGNORECASE,
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def zip_write(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data)


def read_style_file(name: str, style_dir: Path | None) -> bytes:
    if style_dir is not None:
        path = style_dir / name
        if not path.exists():
            raise ValueError(f"missing ACL style file: {path}")
        return path.read_bytes()

    url = f"{ACL_STYLE_BASE}/{name}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "TrajUQ-Overleaf-Packager/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
    except Exception as exc:
        raise ValueError(
            f"failed to fetch pinned ACL style file {name}: {exc}"
        ) from exc
    if not data:
        raise ValueError(f"pinned ACL style file is empty: {name}")
    return data


def scan_text(name: str, data: bytes) -> list[str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [f"{name}: text source is not valid UTF-8"]

    failures: list[str] = []
    for label, pattern in BANNED_PATTERNS.items():
        if pattern.search(text):
            failures.append(f"{name}: contains {label}")
    return failures


def paper_status_text(*, git_commit: str | None) -> bytes:
    commit = git_commit or "UNKNOWN"
    text = f"""# TrajUQ anonymous Overleaf package

This package is generated from the frozen manuscript source.

Source commit: `{commit}`
Official ACL style commit: `{ACL_STYLE_COMMIT}`

The numerical result macros are controlled by `generated_results.tex`.
Before experiments complete, that file intentionally contains TBD placeholders.
After all frozen result artifacts pass their gates, the repository renderer
replaces those placeholders from SHA-bound empirical artifacts.

Do not type empirical numbers manually into `main.tex`.
"""
    return text.encode("utf-8")


def build_package(
    *,
    root: Path,
    output: Path,
    style_dir: Path | None = None,
    git_commit: str | None = None,
) -> dict:
    root = root.resolve()
    paper_dir = root / "paper"
    output = output.resolve()
    if root not in output.parents:
        raise ValueError("output must be inside the repository root")

    files: dict[str, bytes] = {}
    failures: list[str] = []

    for name in PAPER_FILES:
        path = paper_dir / name
        if not path.exists():
            raise ValueError(f"missing paper source: {path}")
        data = path.read_bytes()
        failures.extend(scan_text(name, data))
        files[name] = data

    for name in STYLE_FILES:
        data = read_style_file(name, style_dir)
        # These bytes come only from the pinned official ACL style commit (or
        # an explicit offline style directory in tests). Official acl.sty
        # contains its own acl-org GitHub URL, which is not an anonymity leak.
        files[name] = data

    generated_figures = paper_dir / "figures" / "generated"
    if generated_figures.exists():
        for path in sorted(generated_figures.glob("*.pdf")):
            data = path.read_bytes()
            if not data.startswith(b"%PDF-"):
                raise ValueError(f"invalid generated PDF figure: {path}")
            files[f"figures/generated/{path.name}"] = data

    status = paper_status_text(git_commit=git_commit)
    failures.extend(scan_text("PAPER_STATUS.md", status))
    files["PAPER_STATUS.md"] = status

    if failures:
        raise ValueError(
            "Overleaf anonymity/content scan failed: " + "; ".join(failures)
        )

    manifest = {
        "protocol": "trajuq_overleaf_package_v1",
        "source_git_commit": git_commit,
        "acl_style_repository": "acl-org/acl-style-files",
        "acl_style_commit": ACL_STYLE_COMMIT,
        "anonymous": True,
        "files": [
            {
                "path": name,
                "sha256": sha256_bytes(data),
                "bytes": len(data),
            }
            for name, data in sorted(files.items())
        ],
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w") as zf:
        for name, data in sorted(files.items()):
            zip_write(zf, name, data)
        zip_write(zf, "MANIFEST.json", manifest_bytes)

    archive = output.read_bytes()
    return {
        "protocol": "trajuq_overleaf_package_v1",
        "status": "PASS",
        "output": str(output.relative_to(root)),
        "source_git_commit": git_commit,
        "acl_style_commit": ACL_STYLE_COMMIT,
        "file_count": len(files) + 1,
        "archive_bytes": len(archive),
        "archive_sha256": sha256_bytes(archive),
    }


def git_value(root: Path, *args: str) -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic anonymous Overleaf ZIP from the current "
            "TrajUQ paper source and pinned official ACL style files."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="dist/TrajUQ_NAACL2027_ARR_Overleaf.zip",
    )
    parser.add_argument(
        "--acl-style-dir",
        default=None,
        help=(
            "Optional directory containing acl.sty and acl_natbib.bst. "
            "Used for offline/test builds; otherwise fetch the pinned commit."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    style_dir = (
        Path(args.acl_style_dir).resolve()
        if args.acl_style_dir
        else None
    )
    commit = git_value(root, "rev-parse", "HEAD")
    status = git_value(root, "status", "--porcelain")
    if not commit:
        raise SystemExit("Unable to resolve the source git commit.")
    if status is None:
        raise SystemExit("Unable to resolve git worktree status.")
    if status:
        raise SystemExit(
            "Overleaf packaging requires a clean checkout so the ZIP can be "
            "bound to one source commit."
        )

    try:
        payload = build_package(
            root=root,
            output=output,
            style_dir=style_dir,
            git_commit=commit,
        )
    except ValueError as exc:
        print(
            json.dumps(
                {"status": "FAIL", "reason": str(exc)},
                indent=2,
            )
        )
        raise SystemExit(1) from exc

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
