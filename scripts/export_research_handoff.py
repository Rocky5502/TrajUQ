#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path


CONTEXT_FILES = [
    "PROJECT_STATUS.md",
    "NEW_CHAT_MASTER_PROMPT.md",
    "EXPERIMENT_FREEZE_CHECKLIST.md",
    "configs/project_status.yaml",
    "configs/experiments/primary.yaml",
    "docs/PILOT_RUNBOOK.md",
    "docs/PRIMARY_MATRIX_RUNBOOK.md",
    "docs/A100_EXECUTION.md",
    "docs/RESULT_ARTIFACT_MAP.md",
    "docs/PROGRESS_LOG.md",
    "paper/generated_results.tex",
    "paper/generated_results_manifest.json",
    "paper/figures/generated/result_figure_manifest.json",
]

REQUIRED_CONTEXT_FILES = {
    "PROJECT_STATUS.md",
    "NEW_CHAT_MASTER_PROMPT.md",
    "EXPERIMENT_FREEZE_CHECKLIST.md",
    "configs/project_status.yaml",
    "configs/experiments/primary.yaml",
    "docs/PILOT_RUNBOOK.md",
    "docs/PRIMARY_MATRIX_RUNBOOK.md",
    "docs/A100_EXECUTION.md",
    "docs/RESULT_ARTIFACT_MAP.md",
    "paper/generated_results.tex",
}

RESULT_SUFFIXES = {".json", ".md", ".txt", ".log", ".yaml", ".yml", ".csv"}
SECRET_PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Bearer token": re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def refresh_project_status(root: Path) -> tuple[bool, str]:
    script = root / "scripts" / "project_status.py"
    if not script.exists():
        return False, "scripts/project_status.py is missing"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        tail = completed.stdout[-2000:].strip()
        return False, f"project_status.py failed: {tail}"
    return True, "project status snapshot refreshed"


def collect_files(root: Path, max_file_bytes: int) -> tuple[list[Path], list[str]]:
    selected: set[Path] = set()
    skipped: list[str] = []

    for rel_text in CONTEXT_FILES:
        rel = Path(rel_text)
        path = root / rel
        if path.is_file():
            if path.stat().st_size <= max_file_bytes:
                selected.add(rel)
            else:
                skipped.append(f"{rel.as_posix()}: exceeds per-file byte limit")

    results_root = root / "results"
    if results_root.exists():
        for path in results_root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if path.suffix.lower() not in RESULT_SUFFIXES:
                continue
            if path.stat().st_size > max_file_bytes:
                skipped.append(f"{rel.as_posix()}: exceeds per-file byte limit")
                continue
            selected.add(rel)

    return sorted(selected, key=lambda p: p.as_posix()), sorted(skipped)


def scan_secrets(rel: Path, data: bytes) -> list[str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [f"{rel.as_posix()}: selected metadata file is not valid UTF-8"]

    failures: list[str] = []
    for label, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            failures.append(f"{rel.as_posix()}: contains possible {label}")
    return failures


def zip_write_bytes(zf: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data)


def handoff_summary(commit: str | None, git_clean: bool | None) -> bytes:
    commit_text = commit or "UNKNOWN"
    clean_text = "UNKNOWN" if git_clean is None else ("yes" if git_clean else "no")
    text = f"""# TrajUQ research handoff bundle

Git commit: `{commit_text}`  
Git worktree clean at export: **{clean_text}**

This private handoff archive is for continuing TrajUQ work across chats or machines.
It intentionally includes compact repository context and metadata/result artifacts, but
not raw generation JSONL, benchmark caches, model weights, or files under `runs/`.

## Resume

1. Open `NEW_CHAT_MASTER_PROMPT.md` and paste it into the new chat.
2. Upload this ZIP when local experiment artifacts need to be inspected.
3. Treat `PROJECT_STATUS.md`, `configs/project_status.yaml`, and the latest GitHub
   `main` branch as canonical.
4. Never infer a passed experiment gate from implemented code alone.
"""
    return text.encode("utf-8")


def build_handoff(
    root: Path,
    output: Path,
    *,
    max_file_bytes: int = 2 * 1024 * 1024,
    max_archive_bytes: int = 25 * 1024 * 1024,
) -> dict:
    root = root.resolve()
    output = output.resolve()
    if root not in output.parents:
        raise ValueError("output must be inside the repository root")

    commit = git_value(root, "rev-parse", "HEAD")
    status = git_value(root, "status", "--porcelain")
    git_clean = None if status is None else status == ""

    missing_required = sorted(
        rel_text
        for rel_text in REQUIRED_CONTEXT_FILES
        if not (root / rel_text).is_file()
    )
    if missing_required:
        raise ValueError(
            "required handoff context missing: " + ", ".join(missing_required)
        )

    selected, skipped = collect_files(root, max_file_bytes)
    failures: list[str] = []
    entries: list[dict[str, str | int]] = []
    file_bytes: dict[str, bytes] = {}

    for rel in selected:
        data = (root / rel).read_bytes()
        failures.extend(scan_secrets(rel, data))
        name = rel.as_posix()
        file_bytes[name] = data
        entries.append(
            {
                "path": name,
                "sha256": sha256_bytes(data),
                "bytes": len(data),
            }
        )

    if failures:
        raise ValueError("handoff secret/content scan failed: " + "; ".join(failures))

    summary = handoff_summary(commit, git_clean)
    manifest = {
        "protocol": "trajuq_research_handoff_v1",
        "git_commit": commit,
        "git_clean": git_clean,
        "privacy": {
            "raw_jsonl_included": False,
            "runs_directory_included": False,
            "processed_data_included": False,
            "model_weights_included": False,
            "secret_scan": "PASS",
        },
        "skipped": skipped,
        "files": [
            {
                "path": "HANDOFF_SUMMARY.md",
                "sha256": sha256_bytes(summary),
                "bytes": len(summary),
            },
            *entries,
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    with zipfile.ZipFile(output, "w") as zf:
        zip_write_bytes(zf, "HANDOFF_SUMMARY.md", summary)
        for name in sorted(file_bytes):
            zip_write_bytes(zf, name, file_bytes[name])
        zip_write_bytes(zf, "MANIFEST.json", manifest_bytes)

    archive_bytes = output.stat().st_size
    if archive_bytes > max_archive_bytes:
        output.unlink()
        raise ValueError(
            f"handoff archive exceeds byte limit: {archive_bytes} > {max_archive_bytes}"
        )

    return {
        "protocol": "trajuq_research_handoff_v1",
        "status": "PASS",
        "output": str(output.relative_to(root)),
        "archive_sha256": sha256_bytes(output.read_bytes()),
        "archive_bytes": archive_bytes,
        "included_file_count": len(entries) + 2,
        "skipped_file_count": len(skipped),
        "git_commit": commit,
        "git_clean": git_clean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact private TrajUQ handoff ZIP for a new chat or machine "
            "without raw runs, benchmark caches, or model weights."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default="dist/trajuq_research_handoff.zip",
    )
    parser.add_argument("--max-file-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--max-archive-bytes", type=int, default=25 * 1024 * 1024)
    parser.add_argument(
        "--skip-status-refresh",
        action="store_true",
        help="Do not refresh results/project_status_snapshot.* before packaging.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()

    if not args.skip_status_refresh:
        ok, message = refresh_project_status(root)
        print(f"[TrajUQ] {message}")
        if not ok:
            raise SystemExit(1)

    try:
        payload = build_handoff(
            root,
            output,
            max_file_bytes=args.max_file_bytes,
            max_archive_bytes=args.max_archive_bytes,
        )
    except ValueError as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, indent=2))
        raise SystemExit(1) from exc

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
