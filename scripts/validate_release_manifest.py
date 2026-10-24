#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the reviewer artifact inventory.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("ARTIFACT_MANIFEST.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checked: list[dict[str, str]] = []
    for item in payload.get("artifacts", []):
        rel = item["path"]
        path = root / rel
        if not path.is_file():
            failures.append(f"missing: {rel}")
            continue
        actual = sha256(path)
        expected = item["sha256"].lower()
        if actual != expected:
            failures.append(f"hash mismatch: {rel}")
        checked.append({"path": rel, "sha256": actual, "status": item["status"]})
    report = {
        "status": "PASS" if not failures else "FAIL",
        "experiment_commit": payload.get("experiment_commit"),
        "checked": checked,
        "failures": failures,
        "pending_science": payload.get("pending_science", []),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

