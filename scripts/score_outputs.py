#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from trajuq.io import append_jsonl, read_jsonl, sha256_text
from trajuq.scoring import score_record


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def verify_raw_audit(run_path: Path) -> tuple[Path, dict]:
    audit_path = run_path.with_suffix(run_path.suffix + ".audit.json")
    if not audit_path.exists():
        raise ValueError(f"missing raw-run audit: {audit_path}")
    try:
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid raw-run audit {audit_path}: {exc}") from exc

    commit = git_head()
    if not commit:
        raise ValueError("current git commit could not be resolved for scoring")
    if (
        payload.get("protocol") != "trajuq_run_audit_v1"
        or payload.get("status") != "PASS"
        or payload.get("stage") != "raw"
        or payload.get("git_clean") is not True
        or payload.get("git_commit") != commit
        or str(payload.get("sha256", "")) != sha256_file(run_path)
    ):
        raise ValueError(
            "raw-run audit is not a clean same-commit hash-bound PASS"
        )
    return audit_path, payload


def reference_index(path: Path) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"normalized input is empty: {path}")

    out: dict[str, dict] = {}
    for i, row in enumerate(rows):
        item_id = str(row.get("id", "")).strip()
        if not item_id:
            raise ValueError(f"normalized input row {i} has no id")
        if item_id in out:
            raise ValueError(f"normalized input has duplicate id: {item_id}")
        dataset = str(row.get("dataset", "")).strip()
        if not dataset:
            raise ValueError(f"normalized input row {i} has no dataset identity")
        if row.get("reference") is None:
            raise ValueError(f"normalized input row {i} has no reference")
        out[item_id] = row
    return out


def validate_record_source(
    rec: dict,
    source: dict,
    *,
    item_id: str,
) -> str:
    source_dataset = str(source.get("dataset", "")).strip()
    run_dataset = str(rec.get("dataset", "")).strip()
    if not run_dataset:
        raise ValueError(f"{item_id}: raw run has no dataset identity")
    if run_dataset != source_dataset:
        raise ValueError(
            f"{item_id}: raw-run dataset {run_dataset!r} differs from "
            f"frozen scoring input dataset {source_dataset!r}"
        )

    expected_prompt_hash = sha256_text(str(source["prompt"]))
    if rec.get("prompt_sha256") != expected_prompt_hash:
        raise ValueError(
            f"{item_id}: scoring input prompt differs from generation input"
        )
    return source_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Audited pre-outcome generation JSONL")
    parser.add_argument("--input", required=True, help="Frozen normalized benchmark JSONL with references")
    parser.add_argument("--output", required=True, help="New scored JSONL; never overwrites raw run")
    args = parser.parse_args()

    run_path = Path(args.run)
    input_path = Path(args.input)
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing scored artifact: {out}")

    try:
        refs = reference_index(input_path)
        raw_audit_path, raw_audit = verify_raw_audit(run_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    seen: set[str] = set()
    total = 0
    correct = 0
    raw_sha = sha256_file(run_path)
    raw_audit_sha = sha256_file(raw_audit_path)
    input_sha = sha256_file(input_path)

    for rec in read_jsonl(run_path):
        item_id = str(rec["example_id"])
        if item_id in seen:
            raise SystemExit(f"Duplicate run id: {item_id}")
        seen.add(item_id)

        if rec.get("correct") is not None or rec.get("reference") is not None:
            raise SystemExit(
                f"Refusing to score non-raw artifact: outcome/reference already present for {item_id}"
            )
        if item_id not in refs:
            raise SystemExit(f"Run id missing from normalized input: {item_id}")

        source = refs[item_id]
        try:
            dataset = validate_record_source(
                rec,
                source,
                item_id=item_id,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        is_correct, scorer = score_record(
            dataset=dataset,
            output_text=rec["generation"]["text"],
            reference=str(source["reference"]),
        )

        rec["correct"] = bool(is_correct)
        rec["reference"] = None
        rec.setdefault("metadata", {}).update(
            {
                "scorer": scorer,
                "raw_run_sha256": raw_sha,
                "raw_audit_sha256": raw_audit_sha,
                "raw_audit_git_commit": raw_audit.get("git_commit"),
                "scoring_input_sha256": input_sha,
            }
        )
        append_jsonl(out, rec)
        total += 1
        correct += int(is_correct)

    print(f"[TrajUQ] scored {total} records; correct={correct}; output={out}")


if __name__ == "__main__":
    main()
