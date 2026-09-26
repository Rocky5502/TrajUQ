#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajuq.io import read_jsonl


def get_num_transfer_tokens(torch, mask_index, steps: int):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    counts = torch.zeros(
        mask_num.size(0),
        steps,
        device=mask_index.device,
        dtype=torch.int64,
    ) + base
    for i in range(mask_num.size(0)):
        counts[i, : int(remainder[i].item())] += 1
    return counts


def add_gumbel_noise(torch, logits, temperature: float):
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits64.exp() / gumbel_noise


def native_generate(
    model,
    torch,
    functional,
    prompt,
    attention_mask,
    mask_id: int,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
):
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=model.device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones(
                (prompt.shape[0], gen_length),
                dtype=attention_mask.dtype,
                device=model.device,
            ),
        ],
        dim=-1,
    )
    prompt_index = x.ne(mask_id)

    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        raise ValueError("steps must be divisible by number of blocks")
    per_block = steps // num_blocks

    for block in range(num_blocks):
        start = prompt.shape[1] + block * block_length
        end = prompt.shape[1] + (block + 1) * block_length
        counts = get_num_transfer_tokens(torch, x[:, start:end].eq(mask_id), per_block)

        for i in range(per_block):
            mask_index = x.eq(mask_id)
            if cfg_scale > 0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                joined = torch.cat([x, un_x], dim=0)
                joined_attention = torch.cat([attention_mask, attention_mask], dim=0)
                joined_logits = model(joined, attention_mask=joined_attention).logits
                logits, un_logits = torch.chunk(joined_logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            proposal = torch.argmax(
                add_gumbel_noise(torch, logits, temperature),
                dim=-1,
            )
            if remasking == "low_confidence":
                probs = functional.softmax(logits, dim=-1)
                proposal_prob = probs.gather(-1, proposal.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                proposal_prob = torch.rand(proposal.shape, device=model.device)
            else:
                raise ValueError(remasking)

            proposal_prob[:, end:] = -torch.inf
            proposal = torch.where(mask_index, proposal, x)
            confidence = torch.where(mask_index, proposal_prob, -torch.inf)
            transfer = torch.zeros_like(proposal, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                k = int(counts[j, i].item())
                if k > 0:
                    idx = torch.topk(confidence[j], k=k).indices
                    transfer[j, idx] = True
            x[transfer] = proposal[transfer]
    return x


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Native uninstrumented LLaDA reference generation for parity."
    )
    parser.add_argument("--model", default="inclusionAI/LLaDA-MoE-7B-A1B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", choices=["low_confidence", "random"], default="low_confidence")
    parser.add_argument("--mask-id", type=int, default=None)
    args = parser.parse_args()

    try:
        import torch
        import torch.nn.functional as F
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
    if mask_id is None and "llada" in args.model.lower():
        mask_id = 126336
    if mask_id is None:
        raise SystemExit("Could not infer mask token id; pass --mask-id")

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
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = encoded.input_ids.to(args.device)
            attention_mask = encoded.attention_mask.to(args.device)

            with torch.no_grad():
                output = native_generate(
                    model,
                    torch,
                    F,
                    input_ids,
                    attention_mask,
                    int(mask_id),
                    args.steps,
                    args.max_new_tokens,
                    args.block_length,
                    args.temperature,
                    args.cfg_scale,
                    args.remasking,
                )

            generated = output[0, input_ids.shape[1] :]
            payload = {
                "example_id": str(row["id"]),
                "model": args.model,
                "generation": {
                    "text": tokenizer.decode(
                        generated.tolist(),
                        skip_special_tokens=True,
                    ).strip(),
                    "token_ids": [int(x) for x in generated.tolist()],
                    "metadata": {
                        "model_revision": args.revision,
                        "native_reference": True,
                        "mask_id": int(mask_id),
                        "steps_requested": args.steps,
                    },
                },
            }
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            print(f"[LLaDA reference] {idx + 1}/{len(rows)} {row['id']}")


if __name__ == "__main__":
    main()
