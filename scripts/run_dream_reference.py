#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from trajuq.io import read_jsonl, sha256_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Dream's native diffusion_generate without TrajUQ telemetry for parity."
    )
    parser.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.1)
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit('Install model extras first: pip install -e ".[models]"') from exc

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    dtype_obj = getattr(torch, args.dtype)
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype_obj,
        trust_remote_code=True,
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )

    rows = list(read_jsonl(args.input))[: args.limit]
    with out.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            messages = [{"role": "user", "content": row["prompt"]}]
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
            input_ids = inputs.input_ids.to(args.device)
            attention_mask = inputs.attention_mask.to(args.device)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                output = model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=args.steps,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    alg=args.alg,
                    alg_temp=args.alg_temp,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_time = time.perf_counter() - start
            peak_memory = (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None
            )

            generated = output.sequences[0][input_ids.shape[1] :]
            payload = {
                "example_id": str(row["id"]),
                "dataset": row.get("dataset"),
                "prompt_sha256": sha256_text(str(row["prompt"])),
                "model": args.model,
                "adapter": "dream",
                "seed": int(args.seed),
                "generation": {
                    "text": tokenizer.decode(
                        generated.tolist(),
                        skip_special_tokens=True,
                    ).strip(),
                    "token_ids": [int(x) for x in generated.tolist()],
                    "metadata": {
                        "model_revision": args.revision,
                        "native_reference": True,
                        "steps_requested": int(args.steps),
                        "max_new_tokens": int(args.max_new_tokens),
                        "temperature": float(args.temperature),
                        "top_p": float(args.top_p),
                        "algorithm": args.alg,
                        "algorithm_temperature": float(args.alg_temp),
                        "wall_time_seconds": wall_time,
                        "peak_gpu_memory_bytes": peak_memory,
                        "gpu_name": (
                            torch.cuda.get_device_name()
                            if torch.cuda.is_available()
                            else None
                        ),
                        "torch_version": torch.__version__,
                    },
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            print(f"[Dream reference] {idx + 1}/{len(rows)} {row['id']}")


if __name__ == "__main__":
    main()
