#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Record the human MATH-500 scorer review after manually inspecting "
            "at least 25 frozen calibration examples."
        )
    )
    parser.add_argument(
        "--automatic-audit",
        default="results/math500_scorer_audit.json",
    )
    parser.add_argument("--reviewed-records", type=int, required=True)
    parser.add_argument("--answer-extraction-pass", action="store_true")
    parser.add_argument("--symbolic-equivalence-pass", action="store_true")
    parser.add_argument("--no-manual-label-flips", action="store_true")
    parser.add_argument("--notes", required=True)
    parser.add_argument(
        "--output",
        default="results/math500_scorer_manual_review.json",
    )
    args = parser.parse_args()

    failures: list[str] = []
    audit_path = Path(args.automatic_audit)
    audit_status = None
    if not audit_path.exists():
        failures.append(f"missing automatic scorer audit: {audit_path}")
    else:
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit_status = audit.get("status")
            if audit_status != "PASS":
                failures.append("automatic MATH-500 scorer audit is not PASS")
        except json.JSONDecodeError as exc:
            failures.append(f"invalid automatic audit JSON: {exc}")

    if args.reviewed_records < 25:
        failures.append("at least 25 frozen calibration examples must be reviewed")
    if not args.answer_extraction_pass:
        failures.append("--answer-extraction-pass was not confirmed")
    if not args.symbolic_equivalence_pass:
        failures.append("--symbolic-equivalence-pass was not confirmed")
    if not args.no_manual_label_flips:
        failures.append("--no-manual-label-flips was not confirmed")
    if not args.notes.strip():
        failures.append("manual review notes must be non-empty")

    payload = {
        "protocol": "trajuq_math500_manual_scorer_gate_v1",
        "status": "PASS" if not failures else "FAIL",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "automatic_audit": str(audit_path),
        "automatic_audit_status": audit_status,
        "reviewed_records": int(args.reviewed_records),
        "checks": {
            "answer_extraction": bool(args.answer_extraction_pass),
            "symbolic_equivalence": bool(args.symbolic_equivalence_pass),
            "no_manual_label_flips": bool(args.no_manual_label_flips),
        },
        "notes": args.notes.strip(),
        "failures": failures,
        "attestation": (
            "This artifact records a human calibration-split review. "
            "It must not be used to hand-correct final test labels."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
