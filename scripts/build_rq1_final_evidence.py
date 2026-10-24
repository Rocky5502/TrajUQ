#!/usr/bin/env python
"""Bridge the already frozen RQ1 paper package into the final evidence schema."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "28002055c828c6910840de6f2105aa6367514144"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def repo_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def validate_binding(binding: dict[str, Any], label: str) -> dict[str, str]:
    if set(binding) < {"path", "sha256"}:
        raise ValueError(f"{label} lacks path/SHA-256")
    path = repo_path(str(binding["path"]))
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"{label} is missing or hash-drifted: {binding.get('path')}")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": binding["sha256"]}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary_path = Path(temporary)
        if temporary_path.exists():
            temporary_path.unlink()


def build(source: Path, output: Path) -> dict[str, Any]:
    package = load_json(source)
    if package.get("protocol") != "trajuq_rq1_paper_package_v1":
        raise ValueError("RQ1 paper-package protocol is not recognized")
    if package.get("status") != "PASS" or package.get("authoritative_commit") != EXPECTED_COMMIT:
        raise ValueError("RQ1 paper package is not a PASS at the authoritative commit")

    source_bindings = [validate_binding(item, f"RQ1 source {name}") for name, item in sorted(package["sources"].items())]
    outputs = [validate_binding(package["generated_tex"], "RQ1 generated TeX")]
    outputs.extend(
        validate_binding(item, f"RQ1 figure {name}")
        for name, item in sorted(package["figures"].items())
    )
    primary = load_json(repo_path(package["sources"]["primary"]["path"]))
    gate = load_json(repo_path(package["sources"]["claim_gate"]["path"]))
    if not primary.get("cell_set_complete") or primary.get("observed_cell_count") != 18:
        raise ValueError("RQ1 primary aggregate is not the complete frozen 18-cell set")
    if gate.get("decision") not in {"SUPPORT_POSITIVE_RQ1_CLAIM", "DO_NOT_SUPPORT_POSITIVE_RQ1_CLAIM"}:
        raise ValueError("RQ1 frozen claim gate has no valid decision")
    cell_keys = set(primary.get("per_cell", {}))
    models = ["dream", "llada", "sdar"]
    datasets = ["arc_challenge", "gpqa_diamond", "gsm8k", "math500", "mmlu_pro", "truthfulqa"]
    expected_cells = {f"{model}_{dataset}" for model in models for dataset in datasets}
    if cell_keys != expected_cells:
        raise ValueError("RQ1 identifiers do not resolve to the frozen 3x6 matrix")

    analysis_script = ROOT / "scripts/render_rq1_package.py"
    manifest = {
        "schema": "trajuq_final_rq_evidence_v1",
        "registry_version": "paper-evidence-v1",
        "rq": "RQ1",
        "status": "FINAL",
        "experiment_commit": EXPECTED_COMMIT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "analysis_script": analysis_script.relative_to(ROOT).as_posix(),
        "analysis_script_sha256": sha256_file(analysis_script),
        "audit_status": "PASS",
        "identifiers": {"models": models, "datasets": datasets, "seeds": [int(primary["seed"])]},
        "gates": {
            "generation_complete": True,
            "audits_pass": True,
            "uniqueness_pass": True,
            "completeness_pass": True,
            "scoring_complete": True,
            "analysis_complete": True,
            "claim_gate_decided": True,
            "registry_frozen": True,
        },
        "sources": source_bindings,
        "outputs": outputs,
    }
    atomic_json(output, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/rq1/paper/rq1_paper_manifest.json")
    parser.add_argument("--output", default="results/generated/final/rq1_manifest.json")
    args = parser.parse_args()
    try:
        manifest = build(repo_path(args.source), repo_path(args.output))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "PASS", "rq": manifest["rq"], "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
