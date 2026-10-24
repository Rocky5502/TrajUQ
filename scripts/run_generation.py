#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import time
from importlib import metadata as importlib_metadata
from pathlib import Path

from trajuq.io import append_jsonl, existing_ids, read_jsonl, sha256_text
from trajuq.schema import ExampleRecord


def git_text(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def git_clean() -> bool | None:
    status = git_text("status", "--porcelain")
    return None if status is None else status == ""


def build_adapter(
    name: str,
    model: str,
    device: str,
    dtype: str,
    mask_id: int | None,
    revision: str | None,
):
    if name == "dream":
        from trajuq.adapters.dream import DreamAdapter

        return DreamAdapter(model, device=device, dtype=dtype, revision=revision)
    if name == "llada":
        from trajuq.adapters.llada import LLaDAAdapter

        return LLaDAAdapter(
            model,
            device=device,
            dtype=dtype,
            mask_id=mask_id,
            revision=revision,
        )
    if name == "sdar":
        from trajuq.adapters.sdar import SDARAdapter

        return SDARAdapter(
            model,
            device=device,
            dtype=dtype,
            mask_id=mask_id,
            revision=revision,
        )
    if name == "llama_ar":
        from trajuq.adapters.llama_ar import LlamaARAdapter

        return LlamaARAdapter(
            model,
            device=device,
            dtype=dtype,
            revision=revision,
        )
    if name == "flare":
        raise SystemExit(
            "flare remains an upstream integration boundary until the "
            "HybridDiffusion telemetry/parity gate is complete."
        )
    raise SystemExit(f"Unknown adapter: {name}")


def generate_record(
    adapter,
    args,
    row: dict,
    *,
    seed: int,
    source_git_commit: str | None,
    source_git_clean: bool | None,
    torch,
) -> ExampleRecord:
    """Generate one record using an already-loaded adapter.

    Keeping this function separate lets repeated-sampling runs reuse the same
    resident model across multiple seeds without changing generation semantics
    or the raw-record schema.
    """

    prompt = row["prompt"]
    if torch is not None and args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    if args.adapter == "llama_ar":
        result = adapter.generate(
            prompt,
            seed=seed,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            top_k_telemetry=args.ar_top_k_telemetry,
        )
    elif args.adapter == "dream":
        result = adapter.generate(
            prompt,
            seed=seed,
            steps=args.steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            alg=args.dream_alg,
            alg_temp=args.dream_alg_temp,
        )
    elif args.adapter == "sdar":
        result = adapter.generate(
            prompt,
            seed=seed,
            gen_length=args.max_new_tokens,
            block_length=args.block_length,
            denoising_steps=args.sdar_denoising_steps,
            temperature=(
                args.temperature
                if args.sdar_temperature is None
                else args.sdar_temperature
            ),
            top_k=args.sdar_top_k,
            top_p=args.top_p if args.sdar_top_p is None else args.sdar_top_p,
            remasking_strategy=args.sdar_remasking_strategy,
            confidence_threshold=args.sdar_confidence_threshold,
        )
    else:
        result = adapter.generate(
            prompt,
            seed=seed,
            steps=args.steps,
            gen_length=args.max_new_tokens,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.llada_cfg_scale,
            remasking=args.llada_remasking,
        )

    if torch is not None and args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    result.metadata["wall_time_seconds"] = time.perf_counter() - start
    result.metadata["requested_device"] = str(
        getattr(args, "device", "cuda")
    )
    result.metadata["requested_dtype"] = str(
        getattr(args, "dtype", "bfloat16")
    )
    if torch is not None:
        result.metadata["torch_version"] = torch.__version__
    try:
        result.metadata["transformers_version"] = importlib_metadata.version(
            "transformers"
        )
    except importlib_metadata.PackageNotFoundError:
        result.metadata["transformers_version"] = None
    if torch is not None and args.device.startswith("cuda") and torch.cuda.is_available():
        result.metadata["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        result.metadata["gpu_name"] = torch.cuda.get_device_name()
    result.metadata["pid"] = os.getpid()

    return ExampleRecord(
        example_id=str(row["id"]),
        dataset=str(row.get("dataset", "unknown")),
        prompt_sha256=sha256_text(prompt),
        model=args.model,
        adapter=args.adapter,
        seed=seed,
        generation=result,
        correct=None,
        reference=None,
        metadata={
            "source_input": str(Path(args.input)),
            "model_revision": args.revision,
            "source_git_commit": source_git_commit,
            "source_git_clean": source_git_clean,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        choices=["dream", "llada", "sdar", "llama_ar", "flare"],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None, help="Exact model/tokenizer revision; required for frozen primary runs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--mask-id", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--llada-cfg-scale", type=float, default=0.0)
    parser.add_argument(
        "--llada-remasking",
        choices=["low_confidence", "random"],
        default="low_confidence",
    )
    parser.add_argument(
        "--dream-alg",
        choices=["origin", "maskgit_plus", "topk_margin", "entropy"],
        default="entropy",
    )
    parser.add_argument("--dream-alg-temp", type=float, default=0.1)
    parser.add_argument("--sdar-denoising-steps", type=int, default=4)
    parser.add_argument("--sdar-temperature", type=float, default=None)
    parser.add_argument("--sdar-top-k", type=int, default=0)
    parser.add_argument("--sdar-top-p", type=float, default=None)
    parser.add_argument(
        "--sdar-remasking-strategy",
        choices=["sequential", "low_confidence_static", "low_confidence_dynamic"],
        default="low_confidence_dynamic",
    )
    parser.add_argument("--sdar-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--ar-top-k-telemetry", type=int, default=20)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]

    source_git_commit = git_text("rev-parse", "HEAD")
    source_git_clean = git_clean()

    done = existing_ids(args.output)
    adapter = build_adapter(
        args.adapter,
        args.model,
        args.device,
        args.dtype,
        args.mask_id,
        args.revision,
    )

    torch = None
    try:
        import torch as _torch

        torch = _torch
    except ImportError:
        pass

    for i, row in enumerate(rows):
        example_id = str(row["id"])
        if example_id in done:
            print(f"[TrajUQ] skip completed {example_id}")
            continue

        record = generate_record(
            adapter,
            args,
            row,
            seed=args.seed,
            source_git_commit=source_git_commit,
            source_git_clean=source_git_clean,
            torch=torch,
        )
        append_jsonl(args.output, record.to_dict())
        done.add(example_id)
        print(f"[TrajUQ] {i + 1}/{len(rows)} wrote {example_id}")


if __name__ == "__main__":
    main()
