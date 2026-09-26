#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Record the human pilot-review gate after manual inspection. "
            "This command does not perform the review for you."
        )
    )
    parser.add_argument("--review-sheet", required=True)
    parser.add_argument("--reviewed-records", type=int, required=True)
    parser.add_argument("--trajectory-semantics-pass", action="store_true")
    parser.add_argument("--scorer-audit-pass", action="store_true")
    parser.add_argument("--parity-artifact-reviewed", action="store_true")
    parser.add_argument("--no-leakage-observed", action="store_true")
    parser.add_argument("--notes", required=True)
    parser.add_argument(
        "--output",
        default="results/llada_gsm8k_pilot50_manual_review.json",
    )
    args = parser.parse_args()

    review_sheet = Path(args.review_sheet)
    failures: list[str] = []
    if not review_sheet.exists():
        failures.append(f"missing review sheet: {review_sheet}")
    if args.reviewed_records < 10:
        failures.append("at least 10 trajectory records must be manually reviewed")
    if not args.trajectory_semantics_pass:
        failures.append("--trajectory-semantics-pass was not confirmed")
    if not args.scorer_audit_pass:
        failures.append("--scorer-audit-pass was not confirmed")
    if not args.parity_artifact_reviewed:
        failures.append("--parity-artifact-reviewed was not confirmed")
    if not args.no_leakage_observed:
        failures.append("--no-leakage-observed was not confirmed")
    if not args.notes.strip():
        failures.append("manual review notes must be non-empty")

    payload = {
        "protocol": "trajuq_pilot_manual_gate_v1",
        "status": "PASS" if not failures else "FAIL",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_sheet": str(review_sheet),
        "reviewed_records": int(args.reviewed_records),
        "checks": {
            "trajectory_semantics": bool(args.trajectory_semantics_pass),
            "scorer_audit": bool(args.scorer_audit_pass),
            "parity_artifact_reviewed": bool(args.parity_artifact_reviewed),
            "no_leakage_observed": bool(args.no_leakage_observed),
        },
        "notes": args.notes.strip(),
        "failures": failures,
        "attestation": (
            "This file records a human review; it is not an automated scientific result."
        ),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
