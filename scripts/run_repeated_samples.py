#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


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


def verified_raw_artifact(
    path: Path,
    expected_commit: str | None,
) -> bool:
    audit = path.with_suffix(path.suffix + ".audit.json")
    if not path.exists() or not audit.exists():
        return False
    try:
        payload = json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("protocol") == "trajuq_run_audit_v1"
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
            "clean PASS at the current frozen git commit and its SHA-256 "
            "matches the current artifact."
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
        "runs": [],
    }

    for seed in args.seeds:
        output = out_dir / f"seed-{seed}.raw.jsonl"
        if (
            output.exists()
            and args.resume_verified
            and verified_raw_artifact(output, commit)
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
