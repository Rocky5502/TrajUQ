#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_SEEDS = [2027, 2028, 2029, 2030, 2031]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate frozen repeated-sampling raw artifacts for the UQ baseline."
    )
    parser.add_argument("--adapter", choices=["dream", "llada", "sdar"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--sdar-denoising-steps", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if len(args.seeds) < 2:
        raise SystemExit("Repeated-sampling baseline requires at least two seeds.")
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("Repeated-sampling seeds must be unique.")
    if args.temperature <= 0:
        raise SystemExit(
            "Repeated-sampling baseline requires nonzero temperature; "
            "the frozen primary value is 0.2."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        out = out_dir / f"seed_{seed}.raw.jsonl"
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
            str(out),
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
        if args.adapter == "sdar":
            cmd += [
                "--sdar-denoising-steps",
                str(args.sdar_denoising_steps),
            ]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]

        print("[TrajUQ repeated]", " ".join(cmd))
        subprocess.run(cmd, check=True)
        subprocess.run(
            [
                sys.executable,
                "scripts/audit_run.py",
                "--run",
                str(out),
                "--stage",
                "raw",
                "--require-revision",
            ],
            check=True,
        )

    print(f"[TrajUQ] repeated-sampling raw artifacts ready -> {out_dir}")


if __name__ == "__main__":
    main()
