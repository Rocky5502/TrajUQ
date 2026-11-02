#!/usr/bin/env python
"""Create a non-overwriting snapshot of primary RQ4/RQ5 scientific evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8", errors="replace"
    )


def evidence_paths() -> list[Path]:
    candidates = [
        ROOT / "results/rq4/analysis",
        ROOT / "results/rq4/rq4_downstream_plan.json",
        ROOT / "results/rq4/rq4_downstream_plan.json.sha256",
        ROOT / "results/rq5/analysis",
        ROOT / "results/rq5/census",
        ROOT / "results/rq5/reconciliation",
        ROOT / "results/rq5/validated/registry_manifest.recovered.json",
        ROOT / "results/rq5/validated/registry_manifest.recovered.json.sha256",
        ROOT / "results/rq5/rq5_downstream_plan.json",
        ROOT / "results/rq5/rq5_downstream_plan.json.sha256",
        ROOT / "results/rq5/rq5_downstream_implementation_addendum_v1.json",
        ROOT / "results/rq5/rq5_downstream_implementation_addendum_v2.json",
        ROOT / "results/generated/final/rq4_manifest.json",
        ROOT / "results/generated/final/rq4_scientific_claim_registry.json",
        ROOT / "results/generated/final/rq4_scientific_claim_registry.json.sha256",
        ROOT / "results/generated/final/rq5_manifest.json",
        ROOT / "results/generated/final/rq5_scientific_claim_registry.json",
    ]
    files: set[Path] = set()
    for candidate in candidates:
        if candidate.is_dir():
            files.update(path for path in candidate.rglob("*") if path.is_file())
        elif candidate.is_file():
            files.add(candidate)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite closure snapshot: {output}")
    output.mkdir(parents=True)

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    diff = git("diff", "--binary", "--no-ext-diff")
    cached_diff = git("diff", "--cached", "--binary", "--no-ext-diff")
    provenance = output / "source_state"
    provenance.mkdir()
    state_files = {
        "git_status.txt": status,
        "git_diff.patch": diff,
        "git_diff_cached.patch": cached_diff,
    }
    for name, content in state_files.items():
        (provenance / name).write_text(content, encoding="utf-8", newline="\n")

    artifacts = []
    for source in evidence_paths():
        relative = source.relative_to(ROOT)
        destination = output / "evidence" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        destination_hash = sha256(destination)
        if source_hash != destination_hash:
            raise SystemExit(f"snapshot copy hash mismatch: {relative}")
        artifacts.append(
            {
                "path": relative.as_posix(),
                "sha256": source_hash,
                "bytes": source.stat().st_size,
            }
        )

    manifest = {
        "schema": "trajuq_scientific_closure_snapshot_v1",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_commit": git("rev-parse", "HEAD").strip(),
        "source_branch": git("branch", "--show-current").strip(),
        "dirty": bool(status.strip()),
        "git_status_sha256": sha256(provenance / "git_status.txt"),
        "git_diff_sha256": sha256(provenance / "git_diff.patch"),
        "git_diff_cached_sha256": sha256(provenance / "git_diff_cached.patch"),
        "primary_rq5_result": {
            "estimand": "AURC(verbalized_confidence)-AURC(behavioral_combined)",
            "estimate": -0.002990487543719062,
            "ci_95": [-0.015057308574727653, 0.010202548928115571],
            "holm_adjusted_p": 0.6877312268773123,
            "decision": "NOT_SUPPORTED",
            "immutable": True,
        },
        "artifacts": artifacts,
    }
    manifest_path = output / "scientific_closure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (output / "scientific_closure_manifest.json.sha256").write_text(
        sha256(manifest_path) + "\n", encoding="ascii", newline="\n"
    )
    print(json.dumps({"status": "PASS", "output": str(output), "files": len(artifacts), "manifest_sha256": sha256(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
