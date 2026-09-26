#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def review_sheet_metadata(path: Path) -> dict[str, str]:
    prefixes = {
        "- scored artifact: ": "scored_artifact",
        "- scored SHA-256: ": "scored_sha256",
    }
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        for prefix, key in prefixes.items():
            if line.startswith(prefix):
                found[key] = line[len(prefix):].strip()
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Record the human pilot-review gate after manual inspection. "
            "This command does not perform the review for you."
        )
    )
    parser.add_argument("--review-sheet", required=True)
    parser.add_argument(
        "--parity-artifact",
        default="results/llada_parity.json",
    )
    parser.add_argument(
        "--pilot-quality",
        default="results/llada_gsm8k_pilot50_gate.json",
    )
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
    parity_path = Path(args.parity_artifact)
    quality_path = Path(args.pilot_quality)
    failures: list[str] = []

    review_sheet_sha256 = None
    review_meta: dict[str, str] = {}
    if not review_sheet.exists():
        failures.append(f"missing review sheet: {review_sheet}")
    else:
        review_sheet_sha256 = sha256_file(review_sheet)
        review_meta = review_sheet_metadata(review_sheet)
        if not review_meta.get("scored_artifact") or not review_meta.get("scored_sha256"):
            failures.append("review sheet is missing scored-artifact provenance metadata")

    parity_sha256 = None
    parity_status = None
    parity_protocol = None
    parity: dict = {}
    if not parity_path.exists():
        failures.append(f"missing parity artifact: {parity_path}")
    else:
        try:
            parity = load_json(parity_path)
            parity_status = parity.get("status")
            parity_protocol = parity.get("protocol")
            parity_sha256 = sha256_file(parity_path)
            if parity_status != "PASS":
                failures.append("LLaDA parity artifact is not PASS")
            if parity_protocol != "trajuq_exact_parity_v3":
                failures.append("LLaDA parity artifact is not protocol v3")
        except json.JSONDecodeError as exc:
            failures.append(f"invalid parity artifact JSON: {exc}")

    quality_sha256 = None
    quality_status = None
    quality_protocol = None
    quality: dict = {}
    if not quality_path.exists():
        failures.append(f"missing pilot quality artifact: {quality_path}")
    else:
        try:
            quality = load_json(quality_path)
            quality_status = quality.get("status")
            quality_protocol = quality.get("protocol")
            quality_sha256 = sha256_file(quality_path)
            if quality_status != "PASS":
                failures.append("pilot quality artifact is not PASS")
            if quality_protocol != "trajuq_pilot_quality_gate_v2":
                failures.append("pilot quality artifact is not protocol v2")
        except json.JSONDecodeError as exc:
            failures.append(f"invalid pilot quality artifact JSON: {exc}")

    commit = git_head()
    clean = git_clean()
    if not commit:
        failures.append("git commit could not be resolved")
    if clean is not True:
        failures.append(
            "git worktree must be clean when recording the human attestation"
        )

    if parity:
        if parity.get("git_clean") is not True:
            failures.append("LLaDA parity artifact was not recorded from a clean checkout")
        if commit is not None and parity.get("git_commit") != commit:
            failures.append("LLaDA parity artifact belongs to a different git commit")
        if int(parity.get("shared_records", 0)) != 20:
            failures.append("LLaDA parity artifact does not cover exactly 20 prompts")
        if int(parity.get("mismatch_count", -1)) != 0:
            failures.append("LLaDA parity artifact has output mismatches")
        if int(parity.get("provenance_mismatch_count", -1)) != 0:
            failures.append("LLaDA parity artifact has provenance mismatches")
        for path_key, hash_key, label in (
            ("reference", "reference_sha256", "reference"),
            ("instrumented", "instrumented_sha256", "instrumented"),
        ):
            bound_path = Path(str(parity.get(path_key, "")))
            if (
                not bound_path.exists()
                or parity.get(hash_key) != sha256_file(bound_path)
            ):
                failures.append(f"LLaDA parity {label} source hash is stale")

    if quality:
        if quality.get("git_clean") is not True:
            failures.append("pilot quality artifact was not recorded from a clean checkout")
        if commit is not None and quality.get("git_commit") != commit:
            failures.append("pilot quality artifact belongs to a different git commit")
        quality_scored = Path(str(quality.get("scored", "")))
        quality_scored_sha = str(quality.get("scored_sha256", ""))
        if not quality_scored.exists() or quality_scored_sha != sha256_file(quality_scored):
            failures.append("pilot quality scored source hash is stale")
        if review_meta:
            if review_meta.get("scored_artifact") != str(quality_scored):
                failures.append("review sheet scored artifact differs from pilot quality")
            if review_meta.get("scored_sha256") != quality_scored_sha:
                failures.append("review sheet scored SHA-256 differs from pilot quality")

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
        "protocol": "trajuq_pilot_manual_gate_v2",
        "status": "PASS" if not failures else "FAIL",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_clean": clean,
        "review_sheet": str(review_sheet),
        "review_sheet_sha256": review_sheet_sha256,
        "review_scored_artifact": review_meta.get("scored_artifact"),
        "review_scored_sha256": review_meta.get("scored_sha256"),
        "parity_artifact": str(parity_path),
        "parity_artifact_sha256": parity_sha256,
        "parity_artifact_status": parity_status,
        "parity_artifact_protocol": parity_protocol,
        "pilot_quality": str(quality_path),
        "pilot_quality_sha256": quality_sha256,
        "pilot_quality_status": quality_status,
        "pilot_quality_protocol": quality_protocol,
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
