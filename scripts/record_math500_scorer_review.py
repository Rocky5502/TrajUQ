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


def review_sheet_metadata(path: Path) -> dict[str, str]:
    prefixes = {
        "- scored artifact: ": "scored_artifact",
        "- scored SHA-256: ": "scored_sha256",
        "- normalized input: ": "input",
        "- input SHA-256: ": "input_sha256",
    }
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        for prefix, key in prefixes.items():
            if line.startswith(prefix):
                found[key] = line[len(prefix):].strip()
    return found


def scored_audit_passes(
    path: Path,
    expected_commit: str | None,
) -> tuple[bool, dict]:
    audit_path = path.with_suffix(path.suffix + ".audit.json")
    if not path.exists() or not audit_path.exists():
        return False, {}
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, {}
    passed = (
        payload.get("protocol") == "trajuq_run_audit_v1"
        and payload.get("status") == "PASS"
        and payload.get("git_clean") is True
        and payload.get("git_commit") == expected_commit
        and payload.get("stage") == "scored"
        and str(payload.get("sha256", "")) == sha256_file(path)
    )
    return passed, payload


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
    parser.add_argument(
        "--scored",
        default="runs/llada_math500_scorer_pilot.scored.jsonl",
    )
    parser.add_argument(
        "--input",
        default="data/processed/math500/calibration.jsonl",
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
    review_sheet = Path(args.review_sheet)
    scored_path = Path(args.scored)
    input_path = Path(args.input)

    review_sheet_sha256 = None
    review_meta: dict[str, str] = {}
    if not review_sheet.exists():
        failures.append(f"missing review sheet: {review_sheet}")
    else:
        review_sheet_sha256 = sha256_file(review_sheet)
        review_meta = review_sheet_metadata(review_sheet)
        required_meta = ("scored_artifact", "scored_sha256", "input", "input_sha256")
        if any(not review_meta.get(key) for key in required_meta):
            failures.append("MATH-500 review sheet is missing source provenance metadata")

    scored_sha256 = None
    if not scored_path.exists():
        failures.append(f"missing reviewed scored artifact: {scored_path}")
    else:
        scored_sha256 = sha256_file(scored_path)

    input_sha256 = None
    if not input_path.exists():
        failures.append(f"missing reviewed calibration input: {input_path}")
    else:
        input_sha256 = sha256_file(input_path)

    if review_meta:
        if review_meta.get("scored_artifact") != str(scored_path):
            failures.append("review sheet scored artifact path differs from --scored")
        if scored_sha256 and review_meta.get("scored_sha256") != scored_sha256:
            failures.append("review sheet scored SHA-256 differs from current scored artifact")
        if review_meta.get("input") != str(input_path):
            failures.append("review sheet input path differs from --input")
        if input_sha256 and review_meta.get("input_sha256") != input_sha256:
            failures.append("review sheet input SHA-256 differs from current calibration input")

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
            if audit.get("git_clean") is not True:
                failures.append("automatic MATH-500 scorer audit was not recorded cleanly")
            if int(audit.get("failure_count", 1)) != 0:
                failures.append("automatic MATH-500 scorer audit reports failures")
            dependency_versions = audit.get("dependency_versions", {})
            if dependency_versions.get("math-verify") != "0.9.0":
                failures.append("automatic audit math-verify version drifted")
            if dependency_versions.get("antlr4-python3-runtime") != "4.13.2":
                failures.append("automatic audit antlr runtime version drifted")
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
    scored_audit_ok, _ = scored_audit_passes(scored_path, commit)
    if not scored_audit_ok:
        failures.append(
            "reviewed MATH-500 scored artifact has no current PASS audit at this commit or the audit is not clean"
        )

    normalized_input = str(input_path).replace("\\", "/")
    if not normalized_input.endswith("data/processed/math500/calibration.jsonl"):
        failures.append("MATH-500 human review input must be the frozen calibration split")


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
        "review_scored_artifact": str(scored_path),
        "review_scored_sha256": scored_sha256,
        "review_input": str(input_path),
        "review_input_sha256": input_sha256,
        "review_scored_audit_ok": scored_audit_ok,
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
