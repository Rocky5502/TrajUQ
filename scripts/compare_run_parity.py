#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajuq.io import read_jsonl


def index(path: str) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    out: dict[str, dict] = {}
    for row in rows:
        item_id = str(row.get("example_id", row.get("id", "")))
        if not item_id:
            raise SystemExit(f"{path}: missing example_id/id")
        if item_id in out:
            raise SystemExit(f"{path}: duplicate id {item_id}")
        out[item_id] = row
    return out


def token_ids(row: dict) -> list[int] | None:
    generation = row.get("generation", {})
    ids = generation.get("token_ids")
    if ids is None:
        ids = row.get("token_ids")
    if ids is None:
        return None
    return [int(x) for x in ids]


def generated_text(row: dict) -> str | None:
    generation = row.get("generation", {})
    text = generation.get("text")
    if text is None:
        text = row.get("text")
    return None if text is None else str(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare an upstream/reference generation file with an instrumented TrajUQ run."
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--instrumented", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--require-token-ids", action="store_true")
    args = parser.parse_args()

    ref = index(args.reference)
    inst = index(args.instrumented)
    ref_ids = set(ref)
    inst_ids = set(inst)
    shared = sorted(ref_ids & inst_ids)

    mismatches: list[dict] = []
    missing_reference = sorted(inst_ids - ref_ids)
    missing_instrumented = sorted(ref_ids - inst_ids)

    for item_id in shared:
        a_ids = token_ids(ref[item_id])
        b_ids = token_ids(inst[item_id])
        a_text = generated_text(ref[item_id])
        b_text = generated_text(inst[item_id])

        if args.require_token_ids and (a_ids is None or b_ids is None):
            mismatches.append(
                {
                    "id": item_id,
                    "reason": "missing_token_ids",
                    "reference_has_ids": a_ids is not None,
                    "instrumented_has_ids": b_ids is not None,
                }
            )
            continue

        if a_ids is not None and b_ids is not None:
            if a_ids != b_ids:
                first = next(
                    (i for i, (x, y) in enumerate(zip(a_ids, b_ids)) if x != y),
                    min(len(a_ids), len(b_ids)),
                )
                mismatches.append(
                    {
                        "id": item_id,
                        "reason": "token_mismatch",
                        "first_difference": first,
                        "reference_length": len(a_ids),
                        "instrumented_length": len(b_ids),
                    }
                )
        elif a_text != b_text:
            mismatches.append(
                {
                    "id": item_id,
                    "reason": "text_mismatch",
                }
            )

    payload = {
        "reference": args.reference,
        "instrumented": args.instrumented,
        "shared_records": len(shared),
        "missing_reference": missing_reference,
        "missing_instrumented": missing_instrumented,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
        "status": (
            "PASS"
            if shared
            and not missing_reference
            and not missing_instrumented
            and not mismatches
            else "FAIL"
        ),
    }

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
