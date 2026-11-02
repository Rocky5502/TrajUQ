#!/usr/bin/env python
"""Bind the pre-outcome RQ5 implementation addendum into plan v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    v1_path = ROOT / "results/rq5/rq5_downstream_plan.json"
    v1_hash = sha256(v1_path)
    if v1_hash != Path(str(v1_path) + ".sha256").read_text(encoding="utf-8").strip():
        raise SystemExit("RQ5 v1 plan hash drift")
    addendum_path = ROOT / "results/rq5/rq5_downstream_implementation_addendum_v1.json"
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    if addendum.get("outcomes_inspected") is not False or addendum.get("aggregate_outcomes_computed") is not False:
        raise SystemExit("implementation correction is not pre-outcome")
    plan = json.loads(v1_path.read_text(encoding="utf-8"))
    plan["schema"] = "trajuq_rq5_downstream_plan_v2"
    plan["supersedes_plan_sha256"] = v1_hash
    plan["implementation_addendum"] = {
        "path": addendum_path.relative_to(ROOT).as_posix(),
        "sha256": sha256(addendum_path),
    }
    plan["timing_note"] = (
        "Plan v1 was frozen post-data/pre-outcome. Validation passed; feature extraction failed closed "
        "before output or aggregate outcome computation. Plan v2 binds the execution-only normalized-text "
        "fallback addendum and remains pre-outcome."
    )
    replacements = {
        "scripts/rq5_downstream_stage.py": sha256(ROOT / "scripts/rq5_downstream_stage.py"),
        "scripts/freeze_rq5_downstream_plan.py": None,
    }
    retained = [
        binding for binding in plan["source_bindings"]
        if binding["path"] not in replacements
    ]
    retained.extend(
        [
            {
                "path": "scripts/rq5_downstream_stage.py",
                "sha256": replacements["scripts/rq5_downstream_stage.py"],
            },
            {
                "path": "scripts/freeze_rq5_downstream_plan_v2.py",
                "sha256": sha256(ROOT / "scripts/freeze_rq5_downstream_plan_v2.py"),
            },
            {
                "path": addendum_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(addendum_path),
            },
        ]
    )
    plan["source_bindings"] = sorted(retained, key=lambda row: row["path"])
    plan["input_bindings"].extend(
        [
            {
                "path": v1_path.relative_to(ROOT).as_posix(),
                "sha256": v1_hash,
            },
            {
                "path": addendum_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(addendum_path),
            },
        ]
    )
    plan["eligibility"]["group_prediction"] = (
        "task-canonical answer; parser-unrecognized schema-valid answers use deterministic "
        "casefolded whitespace-normalized raw-answer labels; ties use lowest repeat index"
    )
    output = ROOT / "results/rq5/rq5_downstream_plan_v2.json"
    payload = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(payload)
    with Path(str(output) + ".sha256").open("xb") as handle:
        handle.write((hashlib.sha256(payload).hexdigest() + "\n").encode())


if __name__ == "__main__":
    main()
