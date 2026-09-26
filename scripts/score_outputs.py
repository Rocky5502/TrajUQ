#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from trajuq.io import append_jsonl, read_jsonl
from trajuq.scoring import score_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Pre-outcome generation JSONL")
    parser.add_argument("--input", required=True, help="Normalized benchmark JSONL with references")
    parser.add_argument("--output", required=True, help="New scored JSONL; never overwrites raw run")
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite existing scored artifact: {out}")

    refs = {str(r["id"]): r for r in read_jsonl(args.input)}
    seen: set[str] = set()
    total = 0
    correct = 0

    for rec in read_jsonl(args.run):
        item_id = str(rec["example_id"])
        if item_id in seen:
            raise SystemExit(f"Duplicate run id: {item_id}")
        seen.add(item_id)
        if item_id not in refs:
            raise SystemExit(f"Run id missing from normalized input: {item_id}")

        source = refs[item_id]
        dataset = str(rec["dataset"])
        is_correct, scorer = score_record(
            dataset=dataset,
            output_text=rec["generation"]["text"],
            reference=str(source["reference"]),
        )

        rec["correct"] = bool(is_correct)
        rec["reference"] = None
        rec.setdefault("metadata", {})["scorer"] = scorer
        append_jsonl(out, rec)
        total += 1
        correct += int(is_correct)

    print(f"[TrajUQ] scored {total} records; correct={correct}; output={out}")


if __name__ == "__main__":
    main()
