#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from trajuq.io import append_jsonl, existing_ids, read_jsonl, sha256_text
from trajuq.schema import ExampleRecord


def build_adapter(name: str, model: str, device: str, dtype: str, mask_id: int | None):
    if name == "dream":
        from trajuq.adapters.dream import DreamAdapter

        return DreamAdapter(model, device=device, dtype=dtype)
    if name == "llada":
        from trajuq.adapters.llada import LLaDAAdapter

        return LLaDAAdapter(model, device=device, dtype=dtype, mask_id=mask_id)
    if name in {"sdar", "flare"}:
        raise SystemExit(
            f"{name} is intentionally an upstream integration boundary. "
            "Instrument/parity-test the pinned upstream runtime before primary runs."
        )
    raise SystemExit(f"Unknown adapter: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", choices=["dream", "llada", "sdar", "flare"], required=True)
    parser.add_argument("--model", required=True)
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
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]

    done = existing_ids(args.output)
    adapter = build_adapter(args.adapter, args.model, args.device, args.dtype, args.mask_id)

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

        prompt = row["prompt"]
        if torch is not None and args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        if args.adapter == "dream":
            result = adapter.generate(
                prompt,
                seed=args.seed,
                steps=args.steps,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            result = adapter.generate(
                prompt,
                seed=args.seed,
                steps=args.steps,
                gen_length=args.max_new_tokens,
                block_length=args.block_length,
            )

        if torch is not None and args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

        result.metadata["wall_time_seconds"] = time.perf_counter() - start
        if torch is not None and args.device.startswith("cuda") and torch.cuda.is_available():
            result.metadata["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated())
            result.metadata["gpu_name"] = torch.cuda.get_device_name()
            result.metadata["torch_version"] = torch.__version__
        result.metadata["pid"] = os.getpid()

        record = ExampleRecord(
            example_id=example_id,
            dataset=str(row.get("dataset", "unknown")),
            prompt_sha256=sha256_text(prompt),
            model=args.model,
            adapter=args.adapter,
            seed=args.seed,
            generation=result,
            correct=None,
            reference=None,
            metadata={"source_input": str(Path(args.input))},
        )
        append_jsonl(args.output, record.to_dict())
        done.add(example_id)
        print(f"[TrajUQ] {i + 1}/{len(rows)} wrote {example_id}")


if __name__ == "__main__":
    main()
