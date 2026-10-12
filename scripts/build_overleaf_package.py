#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path


TMLR_STYLE_COMMIT = "7bf90efe3a0debbba703c05c43f3ff7e4d4a2992"
TMLR_STYLE_BASE = (
    "https://raw.githubusercontent.com/JmlrOrg/tmlr-style-file/"
    + TMLR_STYLE_COMMIT
)
STYLE_FILES = ("tmlr.sty", "tmlr.bst", "fancyhdr.sty")
PAPER_FILES = (
    "main.tex",
    "appendix.tex",
    "references.bib",
    "generated_results.tex",
)
RESULTS_MANIFEST = "generated_results_manifest.json"
FIGURE_MANIFEST = "figures/generated/result_figure_manifest.json"
REQUIRED_RESULT_FIGURES = {
    "primary_auroc_matrix",
    "family_macro_auroc",
    "adaptive_tradeoff",
}
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
            raise ValueError(f"missing TMLR style file: {path}")
        return path.read_bytes()

    url = f"{TMLR_STYLE_BASE}/{name}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "TrajUQ-Overleaf-Packager/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
    except Exception as exc:
        raise ValueError(
            f"failed to fetch pinned TMLR style file {name}: {exc}"
        ) from exc
    if not data:
        raise ValueError(f"pinned TMLR style file is empty: {name}")
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


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON manifest: {path}: {exc}") from exc


def submission_result_files(paper_dir: Path) -> dict[str, bytes]:
    """Return only manifest-bound generated figures for a final submission ZIP."""

    generated_results = paper_dir / "generated_results.tex"
    results_manifest_path = paper_dir / RESULTS_MANIFEST
    figure_manifest_path = paper_dir / FIGURE_MANIFEST
    if not results_manifest_path.is_file():
        raise ValueError(
            f"missing submission results manifest: {results_manifest_path}"
        )
    if not figure_manifest_path.is_file():
        raise ValueError(
            f"missing submission figure manifest: {figure_manifest_path}"
        )

    results_manifest = load_json(results_manifest_path)
    if results_manifest.get("protocol") != "trajuq_paper_results_render_v2":
        raise ValueError("generated-results manifest protocol is not recognized")
    if (
        str(results_manifest.get("output_sha256", ""))
        != sha256_bytes(generated_results.read_bytes())
    ):
        raise ValueError(
            "generated_results.tex hash differs from generated-results manifest"
        )

    generated_text = generated_results.read_text(encoding="utf-8")
    forbidden = (
        "PRE-RESULTS FALLBACK",
        r"\tbd",
        "pre-experiment manuscript",
    )
    remaining = [marker for marker in forbidden if marker in generated_text]
    if remaining:
        raise ValueError(
            "submission-ready generated results contain placeholders: "
            + ", ".join(remaining)
        )

    figure_manifest = load_json(figure_manifest_path)
    if figure_manifest.get("protocol") != "trajuq_paper_result_figures_v2":
        raise ValueError("result-figure manifest protocol is not v2")
    outputs = figure_manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("result-figure manifest outputs is not an object")
    if set(outputs) != REQUIRED_RESULT_FIGURES:
        raise ValueError(
            "result-figure manifest output set differs from frozen contract"
        )

    selected: dict[str, bytes] = {}
    generated_root = (paper_dir / "figures" / "generated").resolve()
    for name in sorted(REQUIRED_RESULT_FIGURES):
        item = outputs[name]
        if not isinstance(item, dict):
            raise ValueError(f"invalid result-figure entry: {name}")
        path = Path(str(item.get("path", "")))
        if not path.is_absolute():
            path = (paper_dir.parent / path).resolve()
        else:
            path = path.resolve()
        if generated_root not in path.parents:
            raise ValueError(
                f"result figure is outside paper/figures/generated: {name}"
            )
        if path.suffix.lower() != ".pdf" or not path.is_file():
            raise ValueError(f"missing/invalid result figure PDF: {name}")
        data = path.read_bytes()
        if not data.startswith(b"%PDF-"):
            raise ValueError(f"invalid generated PDF figure: {path}")
        if sha256_bytes(data) != str(item.get("sha256", "")):
            raise ValueError(f"result figure hash differs from manifest: {name}")
        try:
            expected_bytes = int(item.get("bytes", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid result figure byte count: {name}"
            ) from exc
        if len(data) != expected_bytes:
            raise ValueError(f"result figure byte count differs: {name}")
        selected[f"figures/generated/{path.name}"] = data

    return selected


def paper_status_text(
    *,
    source_snapshot_sha256: str,
    submission_ready: bool,
) -> bytes:
    if submission_ready:
        state = (
            "This is a submission-ready source snapshot. Numerical macros in "
            "`generated_results.tex` and the three generated result figures "
            "were admitted only through their repository SHA-bound manifests."
        )
    else:
        state = (
            "This is a development source snapshot. Before experiments complete, "
            "`generated_results.tex` may intentionally contain TBD placeholders; "
            "the repository renderer replaces them only from gated artifacts."
        )
    text = f"""# TrajUQ anonymous Overleaf package

This package is generated from the frozen anonymous manuscript source.

Anonymous source snapshot SHA-256: `{source_snapshot_sha256}`
Official TMLR style commit: `{TMLR_STYLE_COMMIT}`

{state}

Do not type empirical numbers manually into `main.tex`.
"""
    return text.encode("utf-8")


def build_package(
    *,
    root: Path,
    output: Path,
    style_dir: Path | None = None,
    git_commit: str | None = None,
    submission_ready: bool = False,
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
        # These bytes come only from the pinned official TMLR style commit (or
        # an explicit offline style directory in tests). Official style files
        # may contain their own repository URL, which is not an identity leak.
        files[name] = data

    if submission_ready:
        files.update(submission_result_files(paper_dir))
    else:
        generated_figures = paper_dir / "figures" / "generated"
        if generated_figures.exists():
            for path in sorted(generated_figures.glob("*.pdf")):
                data = path.read_bytes()
                if not data.startswith(b"%PDF-"):
                    raise ValueError(f"invalid generated PDF figure: {path}")
                files[f"figures/generated/{path.name}"] = data

    source_snapshot = hashlib.sha256()
    for name in PAPER_FILES:
        source_snapshot.update(name.encode("utf-8"))
        source_snapshot.update(b"\0")
        source_snapshot.update(files[name])
        source_snapshot.update(b"\0")
    source_snapshot_sha256 = source_snapshot.hexdigest()

    status = paper_status_text(
        source_snapshot_sha256=source_snapshot_sha256,
        submission_ready=submission_ready,
    )
    failures.extend(scan_text("PAPER_STATUS.md", status))
    files["PAPER_STATUS.md"] = status

    if failures:
        raise ValueError(
            "Overleaf anonymity/content scan failed: " + "; ".join(failures)
        )

    manifest = {
        "protocol": "trajuq_tmlr_overleaf_package_v1",
        "source_snapshot_sha256": source_snapshot_sha256,
        "tmlr_style_repository": "JmlrOrg/tmlr-style-file",
        "tmlr_style_commit": TMLR_STYLE_COMMIT,
        "anonymous": True,
        "submission_ready": bool(submission_ready),
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
        "protocol": "trajuq_tmlr_overleaf_package_v1",
        "status": "PASS",
        "output": str(output.relative_to(root)),
        "source_git_commit": git_commit,
        "tmlr_style_commit": TMLR_STYLE_COMMIT,
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
            "TrajUQ paper source and pinned official TMLR style files."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="dist/TrajUQ_TMLR_Overleaf.zip",
    )
    parser.add_argument(
        "--tmlr-style-dir",
        default=None,
        help=(
            "Optional directory containing tmlr.sty, tmlr.bst, and "
            "fancyhdr.sty. "
            "Used for offline/test builds; otherwise fetch the pinned commit."
        ),
    )
    parser.add_argument(
        "--submission-ready",
        action="store_true",
        help=(
            "Require rendered-result and result-figure manifests and package "
            "only their hash-verified final figure PDFs."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    style_dir = (
        Path(args.tmlr_style_dir).resolve()
        if args.tmlr_style_dir
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
            submission_ready=args.submission_ready,
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
