#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from trajuq.io import read_jsonl
from trajuq.path_disagreement import compare_paths


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def index(path: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in read_jsonl(path):
        item_id = str(row.get("example_id", row.get("id", "")))
        if not item_id:
            raise SystemExit(f"{path}: row without example_id/id")
        if item_id in result:
            raise SystemExit(f"{path}: duplicate id {item_id}")
        result[item_id] = row
    return result


def generation(row: dict) -> tuple[str, list[int] | None]:
    gen = row.get("generation", row)
    text = str(gen.get("text", ""))
    ids = gen.get("token_ids")
    return text, None if ids is None else [int(x) for x in ids]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join frozen AR and diffusion outputs into pre-outcome disagreement diagnostics."
    )
    parser.add_argument("--ar", required=True)
    parser.add_argument("--diffusion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    ar_path = Path(args.ar)
    diff_path = Path(args.diffusion)
    out_path = Path(args.output)
    if out_path.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {out_path}")

    ar = index(args.ar)
    diff = index(args.diffusion)
    if set(ar) != set(diff):
        only_ar = sorted(set(ar) - set(diff))
        only_diff = sorted(set(diff) - set(ar))
        raise SystemExit(
            f"AR/diffusion ID mismatch: only_ar={len(only_ar)}, only_diffusion={len(only_diff)}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ar_hash = sha256_file(ar_path)
    diff_hash = sha256_file(diff_path)

    with out_path.open("w", encoding="utf-8") as f:
        for item_id in sorted(ar):
            ar_row = ar[item_id]
            diff_row = diff[item_id]

            ar_prompt_hash = ar_row.get("prompt_sha256")
            diff_prompt_hash = diff_row.get("prompt_sha256")
            if ar_prompt_hash and diff_prompt_hash and ar_prompt_hash != diff_prompt_hash:
                raise SystemExit(f"Prompt-hash mismatch for {item_id}")

            dataset = args.dataset or ar_row.get("dataset") or diff_row.get("dataset") or "unknown"
            ar_text, ar_ids = generation(ar_row)
            diff_text, diff_ids = generation(diff_row)
            diagnostic = compare_paths(
                example_id=item_id,
                dataset=str(dataset),
                ar_text=ar_text,
                diffusion_text=diff_text,
                ar_token_ids=ar_ids,
                diffusion_token_ids=diff_ids,
            ).to_dict()
            diagnostic["metadata"].update(
                {
                    "ar_source_sha256": ar_hash,
                    "diffusion_source_sha256": diff_hash,
                    "prompt_sha256": ar_prompt_hash or diff_prompt_hash,
                    "contains_correctness_label": False,
                }
            )
            f.write(json.dumps(diagnostic, ensure_ascii=False, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "status": "PASS",
                "examples": len(ar),
                "output": str(out_path),
                "ar_sha256": ar_hash,
                "diffusion_sha256": diff_hash,
                "warning": "disagreement is diagnostic only; neither path is ground truth",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
