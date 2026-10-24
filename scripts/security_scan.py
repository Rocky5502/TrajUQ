#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{24,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "openai_style_key": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    "bearer_token": re.compile(r"Authorization\s*:\s*Bearer\s+[A-Za-z0-9._~+/=-]{20,}", re.I),
}

BANNED_BASENAMES = {
    ".env",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "credentials.yaml",
    "credentials.yml",
}
BANNED_SUFFIXES = {".pem", ".p12", ".pfx", ".key"}
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".xz",
    ".pt", ".pth", ".bin", ".safetensors", ".ckpt",
}


def tracked_files(root: Path) -> list[Path]:
    try:
        out = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        return [root / item.decode("utf-8") for item in out.split(b"\0") if item]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]


def looks_like_template(path: Path) -> bool:
    name = path.name.lower()
    return "template" in name or "example" in name or "sample" in name


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan tracked TrajUQ files for obvious secret material.")
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.repo.resolve()

    findings: list[str] = []
    for path in tracked_files(root):
        if not path.exists():
            continue
        rel = path.relative_to(root).as_posix()
        lower = path.name.lower()

        if not looks_like_template(path):
            if lower in BANNED_BASENAMES or path.suffix.lower() in BANNED_SUFFIXES:
                findings.append(f"banned credential-like tracked file: {rel}")

        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label} pattern: {rel}")

    if findings:
        print("SECURITY_SCAN=FAIL")
        for finding in sorted(set(findings)):
            print(f"- {finding}")
        return 1

    print("SECURITY_SCAN=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
