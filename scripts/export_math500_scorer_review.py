#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict, deque
from pathlib import Path

from trajuq.io import read_jsonl
from trajuq.scoring import extract_final_field


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_key(seed: int, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()


def choose_diverse(rows: list[dict], count: int, seed: int) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        source = row["source"]
        metadata = source.get("metadata", {})
        subject = str(metadata.get("subject") or "unknown")
        level = str(metadata.get("level") or "unknown")
        groups[(subject, level)].append(row)

    queues: dict[tuple[str, str], deque[dict]] = {}
    for key, items in groups.items():
        ordered = sorted(items, key=lambda item: stable_key(seed, item["id"]))
        queues[key] = deque(ordered)

    selected: list[dict] = []
    keys = sorted(queues)
    while len(selected) < count and any(queues[key] for key in keys):
        for key in keys:
            if len(selected) >= count:
                break
            if queues[key]:
                selected.append(queues[key].popleft())
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a deterministic, metadata-diverse MATH-500 calibration "
            "scorer-review sheet. This command never changes labels."
        )
    )
    parser.add_argument("--scored", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--output",
        default="results/math500_scorer_review.md",
    )
    args = parser.parse_args()

    if args.count < 25:
        raise SystemExit("MATH-500 manual scorer gate requires at least 25 records.")

    scored_path = Path(args.scored)
    input_path = Path(args.input)
    scored_rows = list(read_jsonl(scored_path))
    source_rows = list(read_jsonl(input_path))
    if not scored_rows or not source_rows:
        raise SystemExit("Scored and source artifacts must be non-empty.")

    source = {str(row["id"]): row for row in source_rows}
    if len(source) != len(source_rows):
        raise SystemExit("Duplicate IDs in MATH-500 source input.")

    merged: list[dict] = []
    seen: set[str] = set()
    for row in scored_rows:
        item_id = str(row["example_id"])
        if item_id in seen:
            raise SystemExit(f"Duplicate scored ID: {item_id}")
        seen.add(item_id)
        if item_id not in source:
            raise SystemExit(f"Scored ID missing from source input: {item_id}")
        source_row = source[item_id]
        if str(row.get("dataset")).lower() != "math500":
            raise SystemExit(f"Non-MATH-500 scored row: {item_id}")
        if str(source_row.get("dataset")).lower() != "math500":
            raise SystemExit(f"Non-MATH-500 source row: {item_id}")
        if row.get("correct") is None:
            raise SystemExit(f"Unscored row: {item_id}")
        scorer = row.get("metadata", {}).get("scorer")
        if scorer != "math_verify_0.9.0_precision6":
            raise SystemExit(
                f"Unexpected scorer for {item_id}: {scorer!r}"
            )
        merged.append({"id": item_id, "scored": row, "source": source_row})

    if len(merged) < args.count:
        raise SystemExit(
            f"Need at least {args.count} scored calibration rows; found {len(merged)}."
        )

    selected = choose_diverse(merged, args.count, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# MATH-500 scorer manual review",
        "",
        (
            "Calibration/development review only. Do not use this sheet to "
            "hand-edit final test labels."
        ),
        "",
        f"- scored artifact: {scored_path}",
        f"- scored SHA-256: {sha256_file(scored_path)}",
        f"- normalized input: {input_path}",
        f"- input SHA-256: {sha256_file(input_path)}",
        f"- deterministic seed: {args.seed}",
        f"- records: {len(selected)}",
        "",
        "For each record, inspect answer extraction and symbolic-equivalence correctness. "
        "Record systematic defects; do not flip individual labels by hand.",
        "",
    ]

    for index, item in enumerate(selected, start=1):
        row = item["scored"]
        source_row = item["source"]
        metadata = source_row.get("metadata", {})
        text = str(row["generation"]["text"])
        final = extract_final_field(text)
        lines += [
            f"## {index}. {item['id']}",
            "",
            f"- subject: {metadata.get('subject')}",
            f"- level: {metadata.get('level')}",
            f"- scorer label: {'CORRECT' if row['correct'] else 'INCORRECT'}",
            f"- extracted FINAL: {final}",
            "",
            "### Problem",
            "",
            str(source_row["prompt"]),
            "",
            "### Reference answer",
            "",
            "~~~text",
            str(source_row["reference"]),
            "~~~",
            "",
            "### Model output",
            "",
            "~~~text",
            text,
            "~~~",
            "",
        ]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[TrajUQ] MATH-500 scorer review sheet -> {out}")


if __name__ == "__main__":
    main()
