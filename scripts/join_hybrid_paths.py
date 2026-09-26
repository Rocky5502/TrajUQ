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


def assert_pre_outcome(row: dict, path: str, item_id: str) -> None:
    if row.get("correct") is not None:
        raise SystemExit(f"{path}: correctness leaked into raw path artifact for {item_id}")
    if row.get("reference") is not None:
        raise SystemExit(f"{path}: reference leaked into raw path artifact for {item_id}")
    metadata = row.get("metadata", {})
    forbidden = {"correct", "reference", "label", "y_error", "is_correct"}
    leaked = sorted(forbidden.intersection(metadata))
    if leaked:
        raise SystemExit(
            f"{path}: outcome-like metadata fields for {item_id}: {leaked}"
        )


def checkpoint_identity(row: dict) -> tuple[str | None, str | None]:
    generation_meta = row.get("generation", {}).get("metadata", {})
    row_meta = row.get("metadata", {})
    model = row.get("model") or generation_meta.get("model_id") or row_meta.get("model_id")
    revision = (
        generation_meta.get("model_revision")
        or row_meta.get("model_revision")
        or row.get("model_revision")
    )
    return (
        None if model is None else str(model),
        None if revision is None else str(revision),
    )


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
    parser.add_argument(
        "--require-checkpoint-metadata",
        action="store_true",
        help="Require both paths to expose identical model id and immutable revision.",
    )
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
            assert_pre_outcome(ar_row, args.ar, item_id)
            assert_pre_outcome(diff_row, args.diffusion, item_id)

            ar_prompt_hash = ar_row.get("prompt_sha256")
            diff_prompt_hash = diff_row.get("prompt_sha256")
            if ar_prompt_hash and diff_prompt_hash and ar_prompt_hash != diff_prompt_hash:
                raise SystemExit(f"Prompt-hash mismatch for {item_id}")

            ar_model, ar_revision = checkpoint_identity(ar_row)
            diff_model, diff_revision = checkpoint_identity(diff_row)
            if args.require_checkpoint_metadata:
                if not ar_model or not diff_model or not ar_revision or not diff_revision:
                    raise SystemExit(
                        f"Missing same-checkpoint metadata for {item_id}: "
                        f"ar=({ar_model},{ar_revision}) diffusion=({diff_model},{diff_revision})"
                    )
            if ar_model and diff_model and ar_model != diff_model:
                raise SystemExit(f"Model-id mismatch for {item_id}: {ar_model} != {diff_model}")
            if ar_revision and diff_revision and ar_revision != diff_revision:
                raise SystemExit(
                    f"Model-revision mismatch for {item_id}: "
                    f"{ar_revision} != {diff_revision}"
                )

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
                    "ar_model_id": ar_model,
                    "diffusion_model_id": diff_model,
                    "ar_model_revision": ar_revision,
                    "diffusion_model_revision": diff_revision,
                    "same_checkpoint_verified": bool(
                        ar_model
                        and diff_model
                        and ar_revision
                        and diff_revision
                        and ar_model == diff_model
                        and ar_revision == diff_revision
                    ),
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
