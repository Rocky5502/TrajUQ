#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def git_clean() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return status.strip() == ""
    except Exception:
        return None


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
    parser.add_argument("--review-sheet", required=True)
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
    review_sheet = Path(args.review_sheet)

    review_sheet_sha256 = None
    if not review_sheet.exists():
        failures.append(f"missing review sheet: {review_sheet}")
    else:
        review_sheet_sha256 = sha256_file(review_sheet)

    audit_status = None
    audit_protocol = None
    audit_commit = None
    audit_sha256 = None
    if not audit_path.exists():
        failures.append(f"missing automatic scorer audit: {audit_path}")
    else:
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit_status = audit.get("status")
            audit_protocol = audit.get("protocol")
            audit_commit = audit.get("git_commit")
            audit_sha256 = sha256_file(audit_path)
            if audit_status != "PASS":
                failures.append("automatic MATH-500 scorer audit is not PASS")
            if audit_protocol != "trajuq_math500_scorer_audit_v2":
                failures.append("automatic MATH-500 scorer audit is not protocol v2")
        except json.JSONDecodeError as exc:
            failures.append(f"invalid automatic audit JSON: {exc}")

    commit = git_head()
    clean = git_clean()
    if not commit:
        failures.append("git commit could not be resolved")
    if clean is not True:
        failures.append(
            "git worktree must be clean when recording the human attestation"
        )
    if audit_commit is not None and commit is not None and audit_commit != commit:
        failures.append(
            f"automatic scorer audit commit {audit_commit} differs from current commit {commit}"
        )

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
        "protocol": "trajuq_math500_manual_scorer_gate_v2",
        "status": "PASS" if not failures else "FAIL",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_clean": clean,
        "automatic_audit": str(audit_path),
        "automatic_audit_sha256": audit_sha256,
        "automatic_audit_status": audit_status,
        "automatic_audit_protocol": audit_protocol,
        "automatic_audit_git_commit": audit_commit,
        "review_sheet": str(review_sheet),
        "review_sheet_sha256": review_sheet_sha256,
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
