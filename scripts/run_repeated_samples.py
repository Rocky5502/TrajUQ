#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from trajuq.io import read_jsonl


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


def repeated_seed_contract(args, seed: int) -> dict:
    """Generation settings that must match before an audited seed can be reused."""

    metadata = {
        "model_revision": str(args.revision),
        "temperature": float(args.temperature),
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

    return {
        "adapter": str(args.adapter),
        "model": str(args.model),
        "seed": int(seed),
        "metadata": metadata,
    }


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
    return True


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


def main() -> None:
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
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume-verified",
        action="store_true",
        help=(
            "Skip an existing seed only when its protocol-v1 raw audit is a "
            "clean PASS at the current frozen git commit, its SHA-256 matches, "
            "and every record matches the requested model/seed/sampler settings."
        ),
    )
    args = parser.parse_args()

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

    manifest = {
        "protocol": "repeated_sampling_v1",
        "git_commit": commit,
        "git_clean": clean,
        "adapter": args.adapter,
        "model": args.model,
        "model_revision": args.revision,
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
                repeated_seed_contract(args, seed),
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
        if output.exists() and not args.overwrite:
            raise SystemExit(
                f"Refusing to overwrite unverified {output}. "
                "Inspect/delete the artifact or use --overwrite intentionally."
            )
        if output.exists():
            output.unlink()

        cmd = [
            sys.executable,
            "scripts/run_generation.py",
            "--adapter",
            args.adapter,
            "--model",
            args.model,
            "--revision",
            args.revision,
            "--input",
            args.input,
            "--output",
            str(output),
            "--seed",
            str(seed),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--steps",
            str(args.steps),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--block-length",
            str(args.block_length),
        ]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]

        if args.adapter == "llada":
            cmd += [
                "--llada-cfg-scale",
                str(args.llada_cfg_scale),
                "--llada-remasking",
                args.llada_remasking,
            ]
        elif args.adapter == "dream":
            cmd += [
                "--dream-alg",
                args.dream_alg,
                "--dream-alg-temp",
                str(args.dream_alg_temp),
            ]
        elif args.adapter == "sdar":
            cmd += [
                "--sdar-denoising-steps",
                str(args.sdar_denoising_steps),
                "--sdar-top-k",
                str(args.sdar_top_k),
                "--sdar-remasking-strategy",
                args.sdar_remasking_strategy,
                "--sdar-confidence-threshold",
                str(args.sdar_confidence_threshold),
            ]

        print("[TrajUQ repeated]", " ".join(cmd))
        subprocess.run(cmd, check=True)
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
            {"seed": seed, "path": str(output), "status": "GENERATED"}
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[TrajUQ] repeated-sampling manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
