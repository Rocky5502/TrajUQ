#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from trajuq.io import read_jsonl, sha256_text


def top_k_logits(torch, logits, k: int):
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    minimum = values[..., -1, None]
    return torch.where(logits < minimum, torch.full_like(logits, float("-inf")), logits)


def top_p_logits(torch, functional, logits, p: float):
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative = torch.cumsum(functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool),
        -1,
        sorted_indices,
        sorted_mask,
    )
    return logits.masked_fill(mask, float("-inf"))


def sample_native(torch, functional, logits, temperature: float, top_k: int, top_p: float):
    if temperature <= 0:
        raise ValueError(
            "Pinned upstream SDAR always samples with torch.multinomial; "
            "temperature must be > 0 for the native parity reference."
        )
    shape = logits.shape[:-1]
    vocab = logits.shape[-1]
    flat = logits.reshape(-1, vocab)
    if temperature != 1.0:
        flat = flat / temperature
    flat = top_k_logits(torch, flat, top_k)
    flat = top_p_logits(torch, functional, flat, top_p)
    probs = functional.softmax(flat, dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    token_prob = torch.gather(probs, -1, token)
    return token.view(*shape), token_prob.view(*shape)


def transfer_counts(torch, block_length: int, steps: int):
    base = block_length // steps
    remainder = block_length % steps
    counts = torch.zeros(steps, dtype=torch.int64) + base
    counts[:remainder] += 1
    return counts


def native_generate(
    model,
    torch,
    functional,
    input_ids,
    mask_id: int,
    gen_length: int,
    block_length: int,
    denoising_steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
    remasking_strategy: str,
    confidence_threshold: float,
    stop_ids: list[int],
):
    from transformers.cache_utils import DynamicCache

    prompt_length = input_ids.shape[1]
    past_key_values = DynamicCache()
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=model.device))
    attention = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
    )
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)

    x = torch.full(
        (1, total_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, :prompt_length] = input_ids
    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    forward_evals = 0
    if prefill_length > 0:
        model(
            x[:, :prefill_length],
            attention_mask=attention[:, :prefill_length, :prefill_length],
            position_ids=position_ids[:, :prefill_length],
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
        )
        forward_evals += 1

    counts = transfer_counts(torch, block_length, denoising_steps)

    for block in range(prefill_blocks, num_blocks):
        block_start = block * block_length
        block_end = (block + 1) * block_length
        cur_x = x[:, block_start:block_end].clone()
        cur_attention = attention[:, block_start:block_end, :block_end]
        cur_position_ids = position_ids[:, block_start:block_end]

        for step in range(denoising_steps + 1):
            mask_index = cur_x.eq(mask_id)
            if mask_index.sum() == 0:
                model(
                    cur_x,
                    attention_mask=cur_attention,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True,
                )
                forward_evals += 1
                break

            logits = model(
                cur_x,
                attention_mask=cur_attention,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
            ).logits
            forward_evals += 1

            proposal, proposal_prob = sample_native(
                torch,
                functional,
                logits,
                temperature,
                top_k,
                top_p,
            )

            if step >= len(counts):
                break

            transfer = torch.zeros_like(proposal, dtype=torch.bool)
            if remasking_strategy == "sequential":
                for j in range(cur_x.shape[0]):
                    masked = mask_index[j].nonzero(as_tuple=True)[0]
                    if not len(masked):
                        raise ValueError("No mask tokens found in current block.")
                    first = int(masked.min().item())
                    k = int(counts[step].item())
                    transfer[j, first : first + k] = True
            else:
                confidence = torch.where(mask_index, proposal_prob, -torch.inf)
                for j in range(confidence.shape[0]):
                    k = int(counts[step].item())
                    if remasking_strategy == "low_confidence_dynamic":
                        high = confidence[j] > confidence_threshold
                        if int(high.sum().item()) >= k:
                            transfer[j] = high
                            continue
                    elif remasking_strategy != "low_confidence_static":
                        raise ValueError(remasking_strategy)
                    if k > 0:
                        idx = torch.topk(confidence[j], k=k).indices
                        transfer[j, idx] = True

            cur_x[transfer] = proposal[transfer]

        x[:, block_start:block_end] = cur_x
        generated = x[:, prompt_length : prompt_length + gen_length]
        if stop_ids and any(bool(generated.eq(stop).any()) for stop in stop_ids):
            break

    return x, forward_evals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pinned-upstream SDAR reference generation without TrajUQ telemetry."
    )
    parser.add_argument("--model", default="JetLM/SDAR-1.7B-Chat")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--remasking-strategy",
        choices=["sequential", "low_confidence_static", "low_confidence_dynamic"],
        default="low_confidence_dynamic",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--mask-id", type=int, default=None)
    args = parser.parse_args()

    if args.temperature <= 0:
        raise SystemExit("Native SDAR parity requires temperature > 0.")
    if args.denoising_steps > args.block_length:
        raise SystemExit("denoising_steps must be <= block_length")

    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    except ImportError as exc:
        raise SystemExit('Install model extras first: pip install -e ".[models]"') from exc

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    dtype_obj = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=dtype_obj,
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )

    mask_id = args.mask_id or tokenizer.mask_token_id
    if mask_id is None and getattr(tokenizer, "mask_token", None):
        ids = tokenizer(tokenizer.mask_token, add_special_tokens=False)["input_ids"]
        mask_id = ids[0] if ids else None
    if mask_id is None:
        mask_id = 151669

    generation_config = GenerationConfig.from_pretrained(
        args.model,
        revision=args.revision,
    )
    eos = generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    stop_ids = [int(eos)] if isinstance(eos, int) else [int(x) for x in (eos or [])]

    rows = list(read_jsonl(args.input))[: args.limit]
    with out.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            tokens = tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            )
            input_ids = tokens["input_ids"].to(args.device)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                output, forward_evals = native_generate(
                    model,
                    torch,
                    F,
                    input_ids,
                    int(mask_id),
                    args.max_new_tokens,
                    args.block_length,
                    args.denoising_steps,
                    args.temperature,
                    args.top_k,
                    args.top_p,
                    args.remasking_strategy,
                    args.confidence_threshold,
                    stop_ids,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_time = time.perf_counter() - start
            peak_memory = (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None
            )

            generated = output[0, input_ids.shape[1] : input_ids.shape[1] + args.max_new_tokens]
            payload = {
                "example_id": str(row["id"]),
                "dataset": row.get("dataset"),
                "prompt_sha256": sha256_text(str(row["prompt"])),
                "model": args.model,
                "adapter": "sdar",
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
                        "upstream_repository": "JetAstra/SDAR",
                        "upstream_commit": "4c2749ba103448f45520e8411533710a1e66574d",
                        "mask_id": int(mask_id),
                        "temperature": float(args.temperature),
                        "top_k": int(args.top_k),
                        "top_p": float(args.top_p),
                        "gen_length": int(args.max_new_tokens),
                        "block_length": int(args.block_length),
                        "denoising_steps": int(args.denoising_steps),
                        "remasking_strategy": args.remasking_strategy,
                        "confidence_threshold": float(args.confidence_threshold),
                        "model_forward_evaluations": int(forward_evals),
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
            print(f"[SDAR reference] {idx + 1}/{len(rows)} {row['id']}")


if __name__ == "__main__":
    main()
