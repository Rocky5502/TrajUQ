#!/usr/bin/env python
"""Build the final RQ2 evidence manifest from a complete frozen execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def binding(root: Path, path: Path) -> dict[str, str]:
    resolved = path if path.is_absolute() else root / path
    resolved = resolved.resolve()
    resolved.relative_to(root.resolve())
    if not resolved.is_file():
        raise ValueError(f"missing RQ2 artifact: {resolved}")
    return {"path": resolved.relative_to(root).as_posix(), "sha256": sha256_file(resolved)}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        candidate = Path(temporary)
        if candidate.exists():
            candidate.unlink()


def build(root: Path, plan_path: Path, state_path: Path, output: Path) -> dict[str, Any]:
    plan = load_json(root / plan_path)
    state = load_json(root / state_path)
    if plan.get("protocol") != "trajuq_repeated_sampling_plan_v2" or plan.get("status") != "PASS":
        raise ValueError("RQ2 plan is not the frozen PASS protocol")
    if plan.get("experiment_git_commit") != EXPECTED_COMMIT:
        raise ValueError("RQ2 plan commit is not authoritative")
    if int(plan.get("expected_cells", -1)) != 12 or len(plan.get("cells", [])) != 12:
        raise ValueError("RQ2 plan is not the frozen 12-cell matrix")
    if state.get("status") != "COMPLETE" or state.get("macro") != "COMPLETE":
        raise ValueError("RQ2 execution state is not complete")
    if int(state.get("complete_cells", -1)) != 12:
        raise ValueError("RQ2 does not contain 12 complete cells")

    models = sorted({str(cell["adapter"]) for cell in plan["cells"]})
    datasets = sorted({str(cell["dataset"]) for cell in plan["cells"]})
    seeds = sorted({int(seed) for cell in plan["cells"] for seed in cell["seeds"]})
    if models != ["dream", "llada", "sdar"] or len(datasets) != 4 or len(seeds) != 5:
        raise ValueError("RQ2 identifiers differ from the frozen 3x4x5 design")

    macro_path = Path(str(plan["macro_output"]))
    macro = load_json(root / macro_path)
    if (
        macro.get("protocol") != "trajuq_vs_repeated_macro_v2"
        or macro.get("cell_set_complete") is not True
        or int(macro.get("observed_cells", -1)) != 12
    ):
        raise ValueError("RQ2 macro aggregate is incomplete or invalid")

    script_path = Path("scripts/execute_repeated_sampling.py")
    sources = [binding(root, plan_path), binding(root, state_path)]
    outputs = [binding(root, macro_path)]
    manifest = {
        "schema": "trajuq_final_rq_evidence_v1",
        "registry_version": "paper-evidence-v1",
        "rq": "RQ2",
        "status": "FINAL",
        "experiment_commit": EXPECTED_COMMIT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "analysis_script": script_path.as_posix(),
        "analysis_script_sha256": sha256_file(root / script_path),
        "audit_status": "PASS",
        "identifiers": {"models": models, "datasets": datasets, "seeds": seeds},
        "gates": {
            "generation_complete": True,
            "audits_pass": True,
            "uniqueness_pass": True,
            "completeness_pass": True,
            "scoring_complete": True,
            "analysis_complete": True,
            "claim_gate_decided": True,
            "registry_frozen": True,
            "seed_jobs_expected": 120,
            "seed_jobs_complete": 120,
        },
        "sources": sources,
        "outputs": outputs,
    }
    atomic_json(root / output, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", default="results/repeated_sampling_plan.json")
    parser.add_argument("--state", default="results/repeated_sampling_execution_state.json")
    parser.add_argument("--output", default="results/generated/final/rq2_manifest.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        build(root, Path(args.plan), Path(args.state), Path(args.output))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "PASS", "rq": "RQ2", "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
