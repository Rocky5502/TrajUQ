#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_generation import build_adapter, generate_record  # noqa: E402
from trajuq.io import append_jsonl, existing_ids, read_jsonl  # noqa: E402


DEFAULT_SEEDS = [2027, 2028, 2029, 2030, 2031]


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


def expected_input_ids(path: Path, limit: int | None = None) -> list[str]:
    rows = list(read_jsonl(path))
    if limit is not None:
        rows = rows[:limit]
    ids = [str(row.get("id", "")) for row in rows]
    if not rows or any(not item_id for item_id in ids):
        raise ValueError(f"invalid repeated-sampling input IDs: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate repeated-sampling input IDs: {path}")
    return ids


def artifact_ids_complete(path: Path, expected_ids: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        rows = list(read_jsonl(path))
    except Exception:
        return False
    ids = [str(row.get("example_id", "")) for row in rows]
    return (
        bool(ids)
        and all(ids)
        and len(ids) == len(set(ids))
        and ids == expected_ids
    )


def repeated_seed_contract(
    args,
    seed: int,
    expected_commit: str | None = None,
) -> dict:
    """Generation settings that must match before an audited seed can be reused."""

    metadata = {
        "model_revision": str(args.revision),
        "temperature": float(args.temperature),
        "requested_device": str(getattr(args, "device", "cuda")),
        "requested_dtype": str(getattr(args, "dtype", "bfloat16")),
    }
    if args.adapter == "llada":
        metadata.update(
            {
                "steps_requested": int(args.steps),
                "gen_length": int(args.max_new_tokens),
                "block_length": int(args.block_length),
                "cfg_scale": float(args.llada_cfg_scale),
                "remasking": str(args.llada_remasking),
            }
        )
    elif args.adapter == "dream":
        metadata.update(
            {
                "steps_requested": int(args.steps),
                "max_new_tokens": int(args.max_new_tokens),
                "top_p": float(args.top_p),
                "algorithm": str(args.dream_alg),
                "algorithm_temperature": float(args.dream_alg_temp),
            }
        )
    elif args.adapter == "sdar":
        metadata.update(
            {
                "gen_length": int(args.max_new_tokens),
                "block_length": int(args.block_length),
                "top_p": float(args.top_p),
                "denoising_steps": int(args.sdar_denoising_steps),
                "top_k": int(args.sdar_top_k),
                "remasking_strategy": str(args.sdar_remasking_strategy),
                "confidence_threshold": float(args.sdar_confidence_threshold),
            }
        )
    else:
        raise ValueError(f"unsupported repeated adapter: {args.adapter}")

    contract = {
        "adapter": str(args.adapter),
        "model": str(args.model),
        "seed": int(seed),
        "metadata": metadata,
    }
    if expected_commit is not None:
        contract["record_metadata"] = {
            "source_git_commit": str(expected_commit),
            "source_git_clean": True,
        }
    return contract


def artifact_generation_contract_matches(path: Path, expected: dict) -> bool:
    try:
        rows = list(read_jsonl(path))
    except Exception:
        return False
    if not rows:
        return False

    for row in rows:
        try:
            top_level_matches = (
                str(row.get("adapter", "")) == str(expected["adapter"])
                and str(row.get("model", "")) == str(expected["model"])
                and int(row.get("seed", -1)) == int(expected["seed"])
            )
        except (TypeError, ValueError):
            return False
        if not top_level_matches:
            return False
        metadata = row.get("generation", {}).get("metadata", {})
        for key, value in expected["metadata"].items():
            observed = metadata.get(key)
            if isinstance(value, float):
                try:
                    if float(observed) != value:
                        return False
                except (TypeError, ValueError):
                    return False
            elif observed != value:
                return False

        record_metadata = expected.get("record_metadata", {})
        observed_record_metadata = row.get("metadata", {})
        if not isinstance(observed_record_metadata, dict):
            return False
        for key, value in record_metadata.items():
            if observed_record_metadata.get(key) != value:
                return False
    return True


def resumable_partial_artifact(
    path: Path,
    expected_ids: list[str],
    expected_contract: dict,
) -> bool:
    """Accept only a contract-matching unaudited prefix for interruption recovery."""

    if not path.exists():
        return False
    audit = path.with_suffix(path.suffix + ".audit.json")
    if audit.exists():
        return False
    try:
        rows = list(read_jsonl(path))
    except Exception:
        return False
    if not rows or len(rows) > len(expected_ids):
        return False
    ids = [str(row.get("example_id", "")) for row in rows]
    if ids != expected_ids[: len(ids)] or len(ids) != len(set(ids)):
        return False
    return artifact_generation_contract_matches(path, expected_contract)


def verified_raw_artifact(
    path: Path,
    expected_commit: str | None,
    expected_ids: list[str],
    expected_contract: dict | None = None,
) -> bool:
    audit = path.with_suffix(path.suffix + ".audit.json")
    if not path.exists() or not audit.exists():
        return False
    try:
        payload = json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    contract_ok = (
        expected_contract is None
        or artifact_generation_contract_matches(path, expected_contract)
    )
    return (
        artifact_ids_complete(path, expected_ids)
        and contract_ok
        and payload.get("protocol") == "trajuq_run_audit_v1"
        and payload.get("status") == "PASS"
        and payload.get("git_clean") is True
        and payload.get("git_commit") == expected_commit
        and payload.get("stage") == "raw"
        and str(payload.get("sha256", "")) == sha256_file(path)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the frozen repeated-sampling uncertainty baseline."
    )
    parser.add_argument("--adapter", choices=["llada", "dream", "sdar"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--llada-cfg-scale", type=float, default=0.0)
    parser.add_argument("--llada-remasking", default="low_confidence")
    parser.add_argument("--dream-alg", default="entropy")
    parser.add_argument("--dream-alg-temp", type=float, default=0.1)
    parser.add_argument("--sdar-denoising-steps", type=int, default=4)
    parser.add_argument("--sdar-top-k", type=int, default=0)
    parser.add_argument("--sdar-remasking-strategy", default="low_confidence_dynamic")
    parser.add_argument("--sdar-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mask-id", type=int, default=None)
    parser.add_argument("--sdar-temperature", type=float, default=None)
    parser.add_argument("--sdar-top-p", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help=(
            "Resume an interrupted seed only when its existing JSONL is an "
            "exact contract-matching prefix (including a complete-but-unaudited "
            "prefix) of the frozen input and has no audit artifact yet. The "
            "completed seed is still audited normally."
        ),
    )
    parser.add_argument(
        "--resume-verified",
        action="store_true",
        help=(
            "Skip an existing seed only when its protocol-v1 raw audit is a "
            "clean PASS at the current frozen git commit, its SHA-256 matches, "
            "and every record matches the requested model/seed/sampler settings."
        ),
    )
    return parser


def run_repeated_generation(
    args,
    *,
    resident_adapter=None,
    torch_module=None,
) -> dict:
    """Execute one frozen repeated-sampling partition.

    A caller may provide an already-loaded adapter so multiple frozen
    dataset/partition jobs for the same checkpoint share one model load.
    Artifact contracts, seed audits, and manifest semantics remain identical
    to standalone execution.
    """

    if args.temperature <= 0:
        raise SystemExit(
            "Repeated-sampling baseline requires temperature > 0 so independent seeds "
            "can produce non-degenerate samples."
        )
    if len(args.seeds) < 2:
        raise SystemExit("Repeated-sampling baseline requires at least two seeds.")
    if len(args.seeds) != len(set(args.seeds)):
        raise SystemExit("Repeated-sampling seeds must be unique.")

    commit = git_head()
    clean = git_clean()
    if not commit:
        raise SystemExit("Repeated-sampling generation requires a resolvable git commit.")
    if clean is not True:
        raise SystemExit(
            "Repeated-sampling generation requires the frozen clean checkout."
        )

    input_path = Path(args.input)
    if not input_path.is_file():
        raise SystemExit(f"Repeated-sampling input is missing: {input_path}")
    try:
        expected_ids = expected_input_ids(input_path, args.limit)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(input_path))
    if args.limit is not None:
        rows = rows[: args.limit]

    adapter = resident_adapter
    torch = torch_module

    manifest = {
        "protocol": "repeated_sampling_v1",
        "git_commit": commit,
        "git_clean": clean,
        "adapter": args.adapter,
        "model": args.model,
        "model_revision": args.revision,
        "requested_device": str(args.device),
        "requested_dtype": str(args.dtype),
        "input": args.input,
        "input_sha256": sha256_file(input_path),
        "seeds": args.seeds,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "steps": args.steps,
        "max_new_tokens": args.max_new_tokens,
        "block_length": args.block_length,
        "adapter_generation": {
            "llada_cfg_scale": args.llada_cfg_scale if args.adapter == "llada" else None,
            "llada_remasking": args.llada_remasking if args.adapter == "llada" else None,
            "dream_alg": args.dream_alg if args.adapter == "dream" else None,
            "dream_alg_temp": args.dream_alg_temp if args.adapter == "dream" else None,
            "sdar_denoising_steps": (
                args.sdar_denoising_steps if args.adapter == "sdar" else None
            ),
            "sdar_top_k": args.sdar_top_k if args.adapter == "sdar" else None,
            "sdar_remasking_strategy": (
                args.sdar_remasking_strategy if args.adapter == "sdar" else None
            ),
            "sdar_confidence_threshold": (
                args.sdar_confidence_threshold if args.adapter == "sdar" else None
            ),
        },
        "runs": [],
    }

    for seed in args.seeds:
        output = out_dir / f"seed-{seed}.raw.jsonl"
        if (
            output.exists()
            and args.resume_verified
            and verified_raw_artifact(
                output,
                commit,
                expected_ids,
                repeated_seed_contract(args, seed, commit),
            )
        ):
            print(f"[TrajUQ repeated] verified seed already complete -> {output}")
            manifest["runs"].append(
                {
                    "seed": seed,
                    "path": str(output),
                    "status": "REUSED_VERIFIED",
                }
            )
            continue

        contract = repeated_seed_contract(args, seed, commit)
        resumed = False
        if output.exists():
            if args.resume_partial and resumable_partial_artifact(
                output,
                expected_ids,
                contract,
            ):
                resumed = True
                print(f"[TrajUQ repeated] resuming validated partial seed -> {output}")
            elif not args.overwrite:
                raise SystemExit(
                    f"Refusing to overwrite unverified {output}. "
                    "Use --resume-partial only for a validated prefix, inspect/delete "
                    "the artifact, or use --overwrite intentionally."
                )
            else:
                output.unlink()

        done = existing_ids(output)
        if len(done) < len(expected_ids) and adapter is None:
            adapter = build_adapter(
                args.adapter,
                args.model,
                args.device,
                args.dtype,
                args.mask_id,
                args.revision,
            )
            if torch is None:
                try:
                    import torch as _torch

                    torch = _torch
                except ImportError:
                    torch = None

        for index, row in enumerate(rows):
            example_id = str(row["id"])
            if example_id in done:
                continue
            if adapter is None:
                raise SystemExit(
                    "Internal error: missing resident adapter for incomplete seed."
                )
            record = generate_record(
                adapter,
                args,
                row,
                seed=seed,
                source_git_commit=commit,
                source_git_clean=clean,
                torch=torch,
            )
            append_jsonl(output, record.to_dict())
            done.add(example_id)
            print(
                f"[TrajUQ repeated] seed={seed} "
                f"{index + 1}/{len(rows)} wrote {example_id}"
            )

        if not artifact_ids_complete(output, expected_ids):
            raise SystemExit(
                f"Repeated seed {seed} did not cover the frozen input exactly."
            )
        subprocess.run(
            [
                sys.executable,
                "scripts/audit_run.py",
                "--run",
                str(output),
                "--stage",
                "raw",
                "--require-revision",
            ],
            check=True,
        )
        manifest["runs"].append(
            {
                "seed": seed,
                "path": str(output),
                "status": "RESUMED_AND_GENERATED" if resumed else "GENERATED",
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[TrajUQ] repeated-sampling manifest -> {manifest_path}")
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    run_repeated_generation(args)


if __name__ == "__main__":
    main()
