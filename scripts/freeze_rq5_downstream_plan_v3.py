#!/usr/bin/env python
"""Bind the calibration-degeneracy implementation addendum into RQ5 plan v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    v2_path = ROOT / "results/rq5/rq5_downstream_plan_v2.json"
    v2_hash = sha256(v2_path)
    if v2_hash != Path(str(v2_path) + ".sha256").read_text(encoding="utf-8").strip():
        raise SystemExit("RQ5 v2 plan hash drift")
    addendum_path = ROOT / "results/rq5/rq5_downstream_implementation_addendum_v2.json"
    addendum = json.loads(addendum_path.read_text(encoding="utf-8"))
    if addendum.get("heldout_outcomes_computed") is not False:
        raise SystemExit("calibration correction was not made before held-out evaluation")
    plan = json.loads(v2_path.read_text(encoding="utf-8"))
    plan["schema"] = "trajuq_rq5_downstream_plan_v3"
    plan["supersedes_plan_sha256"] = v2_hash
    plan["implementation_addendum_v2"] = {
        "path": addendum_path.relative_to(ROOT).as_posix(),
        "sha256": sha256(addendum_path),
    }
    plan["timing_note"] = (
        "Plan v3 preserves v2 science and binds a conservative calibration-degeneracy rule after one "
        "single-class calibration cell was encountered, before any held-out evaluation or aggregate result."
    )
    excluded = {
        "scripts/rq5_downstream_stage.py",
        "scripts/freeze_rq5_downstream_plan_v2.py",
    }
    retained = [binding for binding in plan["source_bindings"] if binding["path"] not in excluded]
    retained.extend(
        [
            {"path": "scripts/rq5_downstream_stage.py", "sha256": sha256(ROOT / "scripts/rq5_downstream_stage.py")},
            {"path": "scripts/freeze_rq5_downstream_plan_v3.py", "sha256": sha256(ROOT / "scripts/freeze_rq5_downstream_plan_v3.py")},
            {"path": addendum_path.relative_to(ROOT).as_posix(), "sha256": sha256(addendum_path)},
        ]
    )
    plan["source_bindings"] = sorted(retained, key=lambda row: row["path"])
    plan["input_bindings"].extend(
        [
            {"path": v2_path.relative_to(ROOT).as_posix(), "sha256": v2_hash},
            {"path": addendum_path.relative_to(ROOT).as_posix(), "sha256": sha256(addendum_path)},
        ]
    )
    plan["features"]["single_class_calibration_rule"] = (
        "retain cell with constant empirical error probability; do not pool, drop, retune, or inspect test labels"
    )
    output = ROOT / "results/rq5/rq5_downstream_plan_v3.json"
    payload = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with output.open("xb") as handle:
        handle.write(payload)
    with Path(str(output) + ".sha256").open("xb") as handle:
        handle.write((hashlib.sha256(payload).hexdigest() + "\n").encode())


if __name__ == "__main__":
    main()
