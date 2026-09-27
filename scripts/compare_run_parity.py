#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from trajuq.io import read_jsonl


REQUIRED_GENERATION_KEYS_BY_ADAPTER = {
    "llada": (
        "model_revision",
        "mask_id",
        "steps_requested",
        "gen_length",
        "block_length",
        "temperature",
        "cfg_scale",
        "remasking",
        "torch_version",
        "transformers_version",
    ),
    "dream": (
        "model_revision",
        "steps_requested",
        "max_new_tokens",
        "temperature",
        "top_p",
        "algorithm",
        "algorithm_temperature",
        "torch_version",
        "transformers_version",
    ),
    "sdar": (
        "model_revision",
        "mask_id",
        "gen_length",
        "block_length",
        "denoising_steps",
        "temperature",
        "top_k",
        "top_p",
        "remasking_strategy",
        "confidence_threshold",
        "upstream_repository",
        "upstream_commit",
        "torch_version",
        "transformers_version",
    ),
}

GENERATION_PROVENANCE_KEYS = (
    "model_revision",
    "mask_id",
    "steps_requested",
    "gen_length",
    "max_new_tokens",
    "block_length",
    "temperature",
    "cfg_scale",
    "remasking",
    "top_p",
    "algorithm",
    "algorithm_temperature",
    "denoising_steps",
    "top_k",
    "remasking_strategy",
    "confidence_threshold",
    "upstream_repository",
    "upstream_commit",
    "torch_version",
    "transformers_version",
)


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


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def environment_provenance_errors(
    run_provenance: dict,
    *,
    commit: str | None,
    clean: bool | None,
    require: bool,
) -> list[dict]:
    errors: list[dict] = []
    if not require:
        return errors
    if not commit:
        errors.append({"reason": "git_commit_unavailable"})
    if clean is not True:
        errors.append(
            {"reason": "git_worktree_not_clean", "git_clean": clean}
        )
    source_commit = run_provenance.get("source_git_commit")
    if commit and source_commit != commit:
        errors.append(
            {
                "reason": "source_git_commit_differs_from_comparator",
                "source_git_commit": source_commit,
                "git_commit": commit,
            }
        )
    if run_provenance.get("source_git_clean") is not True:
        errors.append(
            {
                "reason": "source_git_worktree_not_clean",
                "source_git_clean": run_provenance.get("source_git_clean"),
            }
        )
    return errors


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


def generation_metadata(row: dict) -> dict:
    value = row.get("generation", {}).get("metadata", {})
    return value if isinstance(value, dict) else {}


def record_metadata(row: dict) -> dict:
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def provenance_mismatches(
    item_id: str,
    reference: dict,
    instrumented: dict,
    require: bool,
) -> list[dict]:
    mismatches: list[dict] = []

    top_level_keys = ("dataset", "prompt_sha256", "model", "adapter", "seed")
    for key in top_level_keys:
        left = reference.get(key)
        right = instrumented.get(key)
        if require and (left is None or right is None):
            mismatches.append(
                {
                    "id": item_id,
                    "reason": "missing_provenance",
                    "field": key,
                    "reference": left,
                    "instrumented": right,
                }
            )
            continue
        if left is not None and right is not None and left != right:
            mismatches.append(
                {
                    "id": item_id,
                    "reason": "provenance_mismatch",
                    "field": key,
                    "reference": left,
                    "instrumented": right,
                }
            )

    left_record_meta = record_metadata(reference)
    right_record_meta = record_metadata(instrumented)
    for key in ("source_git_commit", "source_git_clean"):
        left = left_record_meta.get(key)
        right = right_record_meta.get(key)
        if require and (left is None or right is None):
            mismatches.append(
                {
                    "id": item_id,
                    "reason": "missing_provenance",
                    "field": f"metadata.{key}",
                    "reference": left,
                    "instrumented": right,
                }
            )
            continue
        if left is not None and right is not None and left != right:
            mismatches.append(
                {
                    "id": item_id,
                    "reason": "provenance_mismatch",
                    "field": f"metadata.{key}",
                    "reference": left,
                    "instrumented": right,
                }
            )
    if require and (
        left_record_meta.get("source_git_clean") is not True
        or right_record_meta.get("source_git_clean") is not True
    ):
        mismatches.append(
            {
                "id": item_id,
                "reason": "source_checkout_not_clean",
                "field": "metadata.source_git_clean",
                "reference": left_record_meta.get("source_git_clean"),
                "instrumented": right_record_meta.get("source_git_clean"),
            }
        )

    left_meta = generation_metadata(reference)
    right_meta = generation_metadata(instrumented)
    adapter = reference.get("adapter") or instrumented.get("adapter")
    required_generation_keys = set(
        REQUIRED_GENERATION_KEYS_BY_ADAPTER.get(str(adapter), ())
    )
    for key in GENERATION_PROVENANCE_KEYS:
        left = left_meta.get(key)
        right = right_meta.get(key)
        if require and key in required_generation_keys and (
            left is None or right is None
        ):
            mismatches.append(
                {
                    "id": item_id,
                    "reason": "missing_provenance",
                    "field": f"generation.metadata.{key}",
                    "reference": left,
                    "instrumented": right,
                }
            )
            continue
        if left is not None and right is not None and left != right:
            mismatches.append(
                {
                    "id": item_id,
                    "reason": "provenance_mismatch",
                    "field": f"generation.metadata.{key}",
                    "reference": left,
                    "instrumented": right,
                }
            )
    return mismatches


def summarize_run_provenance(
    shared: list[str],
    reference: dict[str, dict],
    instrumented: dict[str, dict],
    require: bool,
) -> tuple[dict, list[dict]]:
    summary: dict[str, object] = {}
    errors: list[dict] = []
    fields = ("dataset", "model", "adapter", "seed")
    for field in fields:
        values = {
            reference[item_id].get(field)
            for item_id in shared
        } | {
            instrumented[item_id].get(field)
            for item_id in shared
        }
        values.discard(None)
        if require and len(values) != 1:
            errors.append(
                {
                    "id": "__run__",
                    "reason": "non_uniform_run_provenance",
                    "field": field,
                    "values": sorted(str(value) for value in values),
                }
            )
        summary[field] = next(iter(values)) if len(values) == 1 else None

    revisions = {
        generation_metadata(reference[item_id]).get("model_revision")
        for item_id in shared
    } | {
        generation_metadata(instrumented[item_id]).get("model_revision")
        for item_id in shared
    }
    revisions.discard(None)
    if require and len(revisions) != 1:
        errors.append(
            {
                "id": "__run__",
                "reason": "non_uniform_run_provenance",
                "field": "generation.metadata.model_revision",
                "values": sorted(str(value) for value in revisions),
            }
        )
    summary["model_revision"] = next(iter(revisions)) if len(revisions) == 1 else None

    for runtime_key in ("torch_version", "transformers_version"):
        runtime_values = {
            generation_metadata(reference[item_id]).get(runtime_key)
            for item_id in shared
        } | {
            generation_metadata(instrumented[item_id]).get(runtime_key)
            for item_id in shared
        }
        runtime_values.discard(None)
        if require and len(runtime_values) != 1:
            errors.append(
                {
                    "id": "__run__",
                    "reason": "non_uniform_run_provenance",
                    "field": f"generation.metadata.{runtime_key}",
                    "values": sorted(str(value) for value in runtime_values),
                }
            )
        summary[runtime_key] = (
            next(iter(runtime_values)) if len(runtime_values) == 1 else None
        )

    source_commits = {
        record_metadata(reference[item_id]).get("source_git_commit")
        for item_id in shared
    } | {
        record_metadata(instrumented[item_id]).get("source_git_commit")
        for item_id in shared
    }
    source_commits.discard(None)
    if require and len(source_commits) != 1:
        errors.append(
            {
                "id": "__run__",
                "reason": "non_uniform_run_provenance",
                "field": "metadata.source_git_commit",
                "values": sorted(str(value) for value in source_commits),
            }
        )
    summary["source_git_commit"] = (
        next(iter(source_commits)) if len(source_commits) == 1 else None
    )

    source_clean_values = {
        record_metadata(reference[item_id]).get("source_git_clean")
        for item_id in shared
    } | {
        record_metadata(instrumented[item_id]).get("source_git_clean")
        for item_id in shared
    }
    source_clean_values.discard(None)
    if require and source_clean_values != {True}:
        errors.append(
            {
                "id": "__run__",
                "reason": "non_uniform_run_provenance",
                "field": "metadata.source_git_clean",
                "values": sorted(str(value) for value in source_clean_values),
            }
        )
    summary["source_git_clean"] = (
        True if source_clean_values == {True} else None
    )
    return summary, errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an upstream/reference generation file with an instrumented "
            "TrajUQ run under exact output and optional provenance constraints."
        )
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--instrumented", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--require-token-ids", action="store_true")
    parser.add_argument(
        "--require-provenance",
        action="store_true",
        help=(
            "Require matching dataset, prompt hash, model, adapter, seed, "
            "immutable model revision, and all shared frozen generation settings."
        ),
    )
    args = parser.parse_args()

    ref = index(args.reference)
    inst = index(args.instrumented)
    ref_ids = set(ref)
    inst_ids = set(inst)
    shared = sorted(ref_ids & inst_ids)

    output_mismatches: list[dict] = []
    provenance_errors: list[dict] = []
    missing_reference = sorted(inst_ids - ref_ids)
    missing_instrumented = sorted(ref_ids - inst_ids)

    for item_id in shared:
        provenance_errors.extend(
            provenance_mismatches(
                item_id,
                ref[item_id],
                inst[item_id],
                require=args.require_provenance,
            )
        )

        a_ids = token_ids(ref[item_id])
        b_ids = token_ids(inst[item_id])
        a_text = generated_text(ref[item_id])
        b_text = generated_text(inst[item_id])

        if args.require_token_ids and (a_ids is None or b_ids is None):
            output_mismatches.append(
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
                    (
                        i
                        for i, (left, right) in enumerate(zip(a_ids, b_ids))
                        if left != right
                    ),
                    min(len(a_ids), len(b_ids)),
                )
                output_mismatches.append(
                    {
                        "id": item_id,
                        "reason": "token_mismatch",
                        "first_difference": first,
                        "reference_length": len(a_ids),
                        "instrumented_length": len(b_ids),
                    }
                )
        elif a_text != b_text:
            output_mismatches.append(
                {
                    "id": item_id,
                    "reason": "text_mismatch",
                }
            )

    run_provenance, run_provenance_errors = summarize_run_provenance(
        shared,
        ref,
        inst,
        require=args.require_provenance,
    )
    provenance_errors.extend(run_provenance_errors)

    commit = git_head()
    clean = git_clean()
    environment_errors = environment_provenance_errors(
        run_provenance,
        commit=commit,
        clean=clean,
        require=args.require_provenance,
    )

    status = (
        "PASS"
        if shared
        and not missing_reference
        and not missing_instrumented
        and not output_mismatches
        and not provenance_errors
        and not environment_errors
        else "FAIL"
    )
    payload = {
        "protocol": "trajuq_exact_parity_v3",
        "git_commit": commit,
        "git_clean": clean,
        "reference": args.reference,
        "reference_sha256": sha256_file(args.reference),
        "instrumented": args.instrumented,
        "instrumented_sha256": sha256_file(args.instrumented),
        "require_token_ids": bool(args.require_token_ids),
        "require_provenance": bool(args.require_provenance),
        "run_provenance": run_provenance,
        "shared_records": len(shared),
        "missing_reference": missing_reference,
        "missing_instrumented": missing_instrumented,
        "mismatch_count": len(output_mismatches),
        "mismatches": output_mismatches[:50],
        "provenance_mismatch_count": len(provenance_errors),
        "provenance_mismatches": provenance_errors[:100],
        "environment_errors": environment_errors,
        "status": status,
    }

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
