#!/usr/bin/env python
"""Fail-closed orchestration for the frozen RQ4 sampled-control analysis.

This command is deliberately unable to inspect or analyze an in-progress
campaign.  It validates the immutable 19,420-row generation manifest and all
owned-file hashes before invoking a separately frozen downstream analysis
plan.  The plan may contain analysis commands only; generation commands are
rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_GENERATION_COMMIT = "5a3993c559128ed40bd9de1c504def687a8f861e"
EXPECTED_ADDENDUM_SHA256 = "ed90a0ddb04d088b5004bb0437b1560e639ad97003a658326acdbeb2d831d9a5"
EXPECTED_QUALIFICATION_SHA256 = "db740a435a73b08f44657d4cc54fd3eb29104d48aac60fbd5fdce3c547861d7d"
EXPECTED_ROWS = 19_420
EXPECTED_SHARDS = 60
EXPECTED_SEEDS = [2027, 2028, 2029, 2030, 2031]
EXPECTED_PARTITION_ROWS = {
    ("gsm8k", "calibration"): 500,
    ("gsm8k", "test"): 500,
    ("math500", "calibration"): 153,
    ("math500", "test"): 347,
    ("gpqa_diamond", "calibration"): 56,
    ("gpqa_diamond", "test"): 142,
    ("arc_challenge", "calibration"): 299,
    ("arc_challenge", "test"): 500,
    ("mmlu_pro", "calibration"): 70,
    ("mmlu_pro", "test"): 500,
    ("truthfulqa", "calibration"): 245,
    ("truthfulqa", "test"): 572,
}
EXPECTED_STAGE_ORDER = [
    "scoring",
    "feature_extraction",
    "calibration_fit",
    "heldout_evaluation",
    "paired_bootstrap",
    "multiplicity_correction",
    "hypothesis_gate",
    "macro_aggregate",
]
FORBIDDEN_COMMAND_FRAGMENTS = (
    "run_generation",
    "run_repeated_samples",
    "resident_generate",
    "qualify_rq4_sampling",
    "freeze_rq4_sampling",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        candidate = Path(temporary)
        if candidate.exists():
            candidate.unlink()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def require_generation_complete(
    generation_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Validate completeness using metadata and hashes, never row outcomes."""
    if git(generation_root, "rev-parse", "HEAD") != EXPECTED_GENERATION_COMMIT:
        raise ValueError("RQ4 generation checkout commit drifted")
    if git(generation_root, "status", "--porcelain"):
        raise ValueError("RQ4 generation checkout is dirty")
    addendum = generation_root / "results/rq4/sampling_freeze_addendum.json"
    qualification = generation_root / "results/rq4/sampling_qualification.json"
    if sha256_file(addendum) != EXPECTED_ADDENDUM_SHA256:
        raise ValueError("RQ4 sampling addendum hash mismatch")
    if sha256_file(qualification) != EXPECTED_QUALIFICATION_SHA256:
        raise ValueError("RQ4 sampling qualification hash mismatch")

    manifest = load_object(manifest_path)
    if (
        manifest.get("schema") != "trajuq_tmlr_rq4_sampling_campaign_audit_v1"
        or manifest.get("status") != "PASS"
        or manifest.get("complete") is not True
        or int(manifest.get("rows", -1)) != EXPECTED_ROWS
        or int(manifest.get("expected_rows", -1)) != EXPECTED_ROWS
        or int(manifest.get("complete_shards", -1)) != EXPECTED_SHARDS
        or int(manifest.get("expected_shards", -1)) != EXPECTED_SHARDS
        or manifest.get("source_commit") != EXPECTED_GENERATION_COMMIT
        or manifest.get("sampling_addendum_sha256") != EXPECTED_ADDENDUM_SHA256
        or manifest.get("sampling_qualification_sha256") != EXPECTED_QUALIFICATION_SHA256
        or manifest.get("failures") != []
        or manifest.get("claim_authorized") is not False
    ):
        raise ValueError("RQ4 generation manifest is not exact-complete and audit-PASS")

    settings = manifest.get("scientific_settings", {})
    if settings != {
        "do_sample": True,
        "max_new_tokens": 512,
        "seeds": EXPECTED_SEEDS,
        "temperature": 0.7,
        "top_k": 0,
        "top_p": 0.95,
    }:
        raise ValueError("RQ4 registered sampling settings drifted")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != EXPECTED_SHARDS:
        raise ValueError("RQ4 final manifest does not contain 60 shard audits")
    shard_keys: set[tuple[str, str, int]] = set()
    shard_files: dict[Path, str] = {}
    for shard in shards:
        if not isinstance(shard, dict) or shard.get("complete") is not True:
            raise ValueError("RQ4 final manifest contains an incomplete shard")
        key = (str(shard.get("dataset")), str(shard.get("split")), int(shard.get("seed", -1)))
        expected_rows = EXPECTED_PARTITION_ROWS.get((key[0], key[1]))
        if key in shard_keys or key[2] not in EXPECTED_SEEDS or expected_rows is None:
            raise ValueError("RQ4 shard ownership is duplicated or unregistered")
        if (
            int(shard.get("rows", -1)) != expected_rows
            or int(shard.get("expected_rows", -2)) != expected_rows
        ):
            raise ValueError("RQ4 shard row count is incomplete")
        shard_path = Path(str(shard.get("path", ""))).resolve()
        shard_sha = str(shard.get("sha256", ""))
        if not shard_sha:
            raise ValueError("RQ4 complete shard has no immutable hash")
        shard_files[shard_path] = shard_sha
        shard_keys.add(key)
    expected_keys = {
        (dataset, split, seed)
        for dataset, split in EXPECTED_PARTITION_ROWS
        for seed in EXPECTED_SEEDS
    }
    if shard_keys != expected_keys:
        raise ValueError("RQ4 shard ledger differs from the frozen 6x2x5 design")

    owned = manifest.get("owned_files")
    if not isinstance(owned, list) or len(owned) != EXPECTED_SHARDS:
        raise ValueError("RQ4 immutable owned-file ledger is not exact-complete")
    paths: set[Path] = set()
    owned_hashes: dict[Path, str] = {}
    for item in owned:
        if not isinstance(item, dict):
            raise ValueError("RQ4 owned-file ledger entry is invalid")
        path = Path(str(item.get("path", ""))).resolve()
        if path in paths or not path.is_file():
            raise ValueError("RQ4 owned file is missing or duplicated")
        if sha256_file(path) != item.get("sha256"):
            raise ValueError(f"RQ4 owned-file hash mismatch: {path}")
        paths.add(path)
        owned_hashes[path] = str(item["sha256"])
    if paths != set(shard_files):
        raise ValueError("RQ4 owned-file ledger differs from audited shard paths")
    if any(shard_files[path] != owned_hashes[path] for path in paths):
        raise ValueError("RQ4 shard and owned-file hashes disagree")
    return manifest


def validate_plan(root: Path, plan_path: Path, manifest_path: Path) -> dict[str, Any]:
    plan = load_object(plan_path)
    if (
        plan.get("schema") != "trajuq_rq4_downstream_plan_v1"
        or plan.get("status") != "PASS"
        or int(plan.get("expected_generation_rows", -1)) != EXPECTED_ROWS
        or int(plan.get("expected_generation_shards", -1)) != EXPECTED_SHARDS
        or plan.get("generation_manifest_sha256") != sha256_file(manifest_path)
        or plan.get("sampling_addendum_sha256") != EXPECTED_ADDENDUM_SHA256
        or plan.get("sampling_qualification_sha256") != EXPECTED_QUALIFICATION_SHA256
    ):
        raise ValueError("RQ4 downstream plan is not frozen to the complete campaign")
    science = plan.get("registered_science", {})
    if science != {
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 2027,
        "confidence_level": 0.95,
        "equal_cell_weight": True,
        "multiplicity_correction": "holm_bonferroni",
        "confirmatory_family": ["h4a", "h4b", "h4c"],
        "two_sided_alpha": 0.05,
    }:
        raise ValueError("RQ4 downstream statistical contract drifted")

    bindings = plan.get("source_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("RQ4 downstream plan has no hash-bound source registry")
    for binding in bindings:
        path = _inside(root, root / str(binding.get("path", "")))
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            raise ValueError(f"RQ4 downstream source binding mismatch: {path}")

    stages = plan.get("stages")
    if (
        not isinstance(stages, list)
        or [stage.get("name") for stage in stages] != EXPECTED_STAGE_ORDER
    ):
        raise ValueError("RQ4 downstream stages are missing, duplicated, or out of order")
    output_paths: set[Path] = set()
    for stage in stages:
        argv = stage.get("argv")
        outputs = stage.get("outputs")
        if (
            not isinstance(argv, list)
            or len(argv) < 2
            or not all(isinstance(item, str) and item for item in argv)
        ):
            raise ValueError(f"invalid argv for RQ4 stage {stage.get('name')}")
        if Path(argv[0]).name.lower() not in {"python", "python3", "python.exe"}:
            raise ValueError(
                f"RQ4 stage must invoke the pinned Python runtime: {stage.get('name')}"
            )
        command_text = " ".join(argv).lower()
        if any(fragment in command_text for fragment in FORBIDDEN_COMMAND_FRAGMENTS):
            raise ValueError(f"generation command forbidden in RQ4 finalizer: {stage.get('name')}")
        script = _inside(root, root / argv[1])
        if (
            script.suffix != ".py"
            or not script.is_file()
            or sha256_file(script) != stage.get("script_sha256")
        ):
            raise ValueError(f"RQ4 stage script binding mismatch: {stage.get('name')}")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError(f"RQ4 stage has no declared outputs: {stage.get('name')}")
        for output in outputs:
            path = _inside(root, root / str(output.get("path", "")))
            if path in output_paths:
                raise ValueError("RQ4 downstream output ownership overlaps")
            output_paths.add(path)
    return plan


def _finite_json(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    return True


def validate_stage_outputs(root: Path, stage: dict[str, Any]) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    for spec in stage["outputs"]:
        path = _inside(root, root / str(spec["path"]))
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"RQ4 stage output missing or empty: {path}")
        if path.suffix == ".json":
            payload = load_object(path)
            if not _finite_json(payload):
                raise ValueError(f"RQ4 stage output contains non-finite JSON: {path}")
            expected_protocol = spec.get("protocol")
            if expected_protocol and payload.get("protocol") != expected_protocol:
                raise ValueError(f"RQ4 stage output protocol mismatch: {path}")
        elif path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                count = 0
                for line in handle:
                    if not line.strip():
                        continue
                    if not _finite_json(json.loads(line)):
                        raise ValueError(f"RQ4 stage output contains non-finite JSONL: {path}")
                    count += 1
                if count == 0:
                    raise ValueError(f"RQ4 stage output has no JSONL records: {path}")
        bindings.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    return bindings


def execute(
    root: Path,
    generation_root: Path,
    generation_manifest_path: Path,
    plan_path: Path,
    state_path: Path,
    final_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    generation = require_generation_complete(generation_root, generation_manifest_path)
    plan = validate_plan(root, plan_path, generation_manifest_path)
    plan_sha = sha256_file(plan_path)
    if final_path.exists():
        raise ValueError("refusing to overwrite immutable RQ4 final evidence manifest")

    state: dict[str, Any] = {
        "schema": "trajuq_rq4_downstream_state_v1",
        "status": "DRY_RUN" if dry_run else "RUNNING",
        "plan_sha256": plan_sha,
        "generation_manifest_sha256": sha256_file(generation_manifest_path),
        "completed_stages": [],
    }
    if state_path.exists():
        state = load_object(state_path)
        if (
            state.get("plan_sha256") != plan_sha
            or state.get("generation_manifest_sha256")
            != sha256_file(generation_manifest_path)
        ):
            raise ValueError("RQ4 downstream resume state provenance mismatch")

    completed = {item["name"]: item for item in state.get("completed_stages", [])}
    for stage in plan["stages"]:
        name = str(stage["name"])
        if name in completed:
            current = validate_stage_outputs(root, stage)
            if current != completed[name].get("outputs"):
                raise ValueError(f"RQ4 completed stage artifact drift: {name}")
            continue
        for output in stage["outputs"]:
            if (root / str(output["path"])).exists():
                raise ValueError(
                    "refusing to overwrite unregistered RQ4 stage output: "
                    f"{output['path']}"
                )
        # Recheck campaign and source integrity immediately before every stage.
        require_generation_complete(generation_root, generation_manifest_path)
        argv = [sys.executable, *stage["argv"][1:]]
        print("[DRY]" if dry_run else "[RUN]", name, " ".join(argv))
        if dry_run:
            continue
        subprocess.run(argv, cwd=root, check=True)
        outputs = validate_stage_outputs(root, stage)
        state.setdefault("completed_stages", []).append({"name": name, "outputs": outputs})
        atomic_json(state_path, state)

    if dry_run:
        return {"status": "PASS", "dry_run": True, "stages": EXPECTED_STAGE_ORDER}
    if [item["name"] for item in state["completed_stages"]] != EXPECTED_STAGE_ORDER:
        raise ValueError("RQ4 downstream chain did not complete in registered order")
    require_generation_complete(generation_root, generation_manifest_path)
    final = {
        "schema": "trajuq_final_rq_evidence_v1",
        "registry_version": "paper-evidence-v1",
        "rq": "RQ4",
        "status": "FINAL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "generation_commit": EXPECTED_GENERATION_COMMIT,
        "audit_status": "PASS",
        "claim_authorized": False,
        "claim_authorization_note": (
            "Mechanical finalization is not a scientific claim decision; use "
            "the registered hypothesis-gate artifact."
        ),
        "gates": {
            "generation_rows_expected": EXPECTED_ROWS,
            "generation_rows_complete": int(generation["rows"]),
            "generation_shards_expected": EXPECTED_SHARDS,
            "generation_shards_complete": int(generation["complete_shards"]),
            "generation_audits_pass": True,
            "provenance_hashes_pass": True,
            "registered_stage_order_complete": True,
            "claim_gate_decided": True,
        },
        "sources": [
            {
                "path": str(generation_manifest_path),
                "sha256": sha256_file(generation_manifest_path),
            },
            {"path": str(plan_path), "sha256": plan_sha},
        ],
        "stage_outputs": state["completed_stages"],
    }
    atomic_json(final_path, final)
    state["status"] = "COMPLETE"
    state["final_manifest"] = {"path": str(final_path), "sha256": sha256_file(final_path)}
    atomic_json(state_path, state)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--generation-root", required=True)
    parser.add_argument("--generation-manifest", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--state", default="results/rq4/analysis/finalization_state.json")
    parser.add_argument("--output", default="results/generated/final/rq4_manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        result = execute(
            root,
            Path(args.generation_root).resolve(),
            Path(args.generation_manifest).resolve(),
            Path(args.plan).resolve(),
            (root / args.state).resolve(),
            (root / args.output).resolve(),
            args.dry_run,
        )
    except (
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {"status": "PASS", "dry_run": args.dry_run, "result": result},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
