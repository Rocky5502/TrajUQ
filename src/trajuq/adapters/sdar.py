from __future__ import annotations

from dataclasses import replace

from trajuq.adapters.base import BaseAdapter
from trajuq.schema import GenerationResult, StepSummary
from trajuq.trajectory import TrajectoryAccumulator


UPSTREAM_REPOSITORY = "JetAstra/SDAR"
UPSTREAM_COMMIT = "4c2749ba103448f45520e8411533710a1e66574d"


def _combine_block_features(
    blocks: list[tuple[int, dict[str, float], list[StepSummary]]],
) -> tuple[dict[str, float], list[StepSummary]]:
    """Combine block-local trajectory summaries into one example record.

    SDAR denoises one block at a time. We therefore compute the same feature
    dictionary within each generated block and aggregate token-weighted values
    across blocks. Extremal features preserve their max/min semantics.
    """

    if not blocks:
        raise ValueError("No generated SDAR blocks were observed.")

    total_tokens = sum(n for n, _, _ in blocks)
    if total_tokens <= 0:
        raise ValueError("Observed SDAR blocks contain no generated positions.")

    feature_names = set(blocks[0][1])
    if any(set(features) != feature_names for _, features, _ in blocks):
        raise ValueError("SDAR block feature schemas do not match.")

    max_features = {"entropy_max", "js_max"}
    min_features = {"margin_min"}
    sum_features = {"trajectory_steps"}
    combined: dict[str, float] = {}

    for name in sorted(feature_names):
        values = [(n, float(features[name])) for n, features, _ in blocks]
        if name in max_features:
            combined[name] = max(v for _, v in values)
        elif name in min_features:
            combined[name] = min(v for _, v in values)
        elif name in sum_features:
            combined[name] = float(sum(v for _, v in values))
        else:
            combined[name] = float(sum(n * v for n, v in values) / total_tokens)

    summaries: list[StepSummary] = []
    global_step = 0
    for _, _, local in blocks:
        for summary in local:
            summaries.append(replace(summary, step=global_step))
            global_step += 1
    return combined, summaries


class SDARAdapter(BaseAdapter):
    """Instrumented adapter for the public SDAR Transformers runtime.

    Generation follows the public JetAstra/SDAR block-diffusion loop at the
    pinned integration revision, with compact telemetry added before sampling.
    Primary-paper use is still gated on an on-GPU parity test against upstream.
    """

    name = "sdar"

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        mask_id: int | None = None,
        revision: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

        self.torch = torch
        self.device = device
        self.model_id = model_id
        self.revision = revision
        dtype_obj = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=dtype_obj,
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        )

        inferred = self.tokenizer.mask_token_id
        if inferred is None and getattr(self.tokenizer, "mask_token", None):
            ids = self.tokenizer(
                self.tokenizer.mask_token,
                add_special_tokens=False,
            )["input_ids"]
            inferred = ids[0] if ids else None
        if mask_id is None:
            mask_id = inferred
        if mask_id is None:
            # SDAR's public README uses this mask id in its JetEngine example.
            mask_id = 151669
        self.mask_id = int(mask_id)

        generation_config = GenerationConfig.from_pretrained(
            model_id,
            revision=revision,
        )
        eos = generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.stop_ids = [int(eos)] if isinstance(eos, int) else [int(x) for x in (eos or [])]

    @staticmethod
    def _top_k_logits(torch, logits, k: int):
        if k <= 0:
            return logits
        values, _ = torch.topk(logits, k)
        minimum = values[..., -1, None]
        return torch.where(logits < minimum, torch.full_like(logits, float("-inf")), logits)

    @staticmethod
    def _top_p_logits(torch, functional, logits, p: float):
        if p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(functional.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(
            torch.full_like(logits, False, dtype=torch.bool),
            -1,
            sorted_indices,
            sorted_mask,
        )
        return logits.masked_fill(mask_indices, float("-inf"))

    def _sample(self, logits, temperature: float, top_k: int, top_p: float):
        torch = self.torch
        import torch.nn.functional as F

        shape = logits.shape[:-1]
        vocab = logits.shape[-1]
        flat = logits.reshape(-1, vocab)

        if temperature <= 0:
            token = flat.argmax(dim=-1, keepdim=True)
            probs = F.softmax(flat, dim=-1)
            token_prob = torch.gather(probs, -1, token)
            return token.view(*shape), token_prob.view(*shape)

        if temperature != 1.0:
            flat = flat / temperature
        flat = self._top_k_logits(torch, flat, top_k)
        flat = self._top_p_logits(torch, F, flat, top_p)
        probs = F.softmax(flat, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_prob = torch.gather(probs, -1, token)
        return token.view(*shape), token_prob.view(*shape)

    @staticmethod
    def _transfer_counts(torch, block_length: int, steps: int, device):
        base = block_length // steps
        remainder = block_length % steps
        counts = torch.full((steps,), base, dtype=torch.long, device=device)
        if remainder:
            counts[:remainder] += 1
        return counts

    def generate(
        self,
        prompt: str,
        seed: int = 2027,
        gen_length: int = 128,
        block_length: int = 4,
        denoising_steps: int = 4,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        remasking_strategy: str = "low_confidence_dynamic",
        confidence_threshold: float = 0.85,
        **_: object,
    ) -> GenerationResult:
        torch = self.torch
        from transformers.cache_utils import DynamicCache

        if block_length <= 0 or denoising_steps <= 0:
            raise ValueError("block_length and denoising_steps must be positive")
        if denoising_steps > block_length:
            raise ValueError("denoising_steps must be <= block_length for SDAR transfer counts")
        if remasking_strategy not in {
            "sequential",
            "low_confidence_static",
            "low_confidence_dynamic",
        }:
            raise ValueError(f"Unsupported SDAR remasking strategy: {remasking_strategy}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        tokens = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = tokens["input_ids"].to(self.device)
        prompt_length = input_ids.shape[1]

        past_key_values = DynamicCache()
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
        attention = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
        )
        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)

        x = torch.full(
            (1, total_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device,
        )
        x[:, :prompt_length] = input_ids

        prefill_blocks = prompt_length // block_length
        prefill_length = prefill_blocks * block_length
        if prefill_length > 0:
            self.model(
                x[:, :prefill_length],
                attention_mask=attention[:, :prefill_length, :prefill_length],
                position_ids=position_ids[:, :prefill_length],
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=True,
            )

        counts = self._transfer_counts(
            torch,
            block_length,
            denoising_steps,
            self.device,
        )
        block_records: list[tuple[int, dict[str, float], list[StepSummary]]] = []
        forward_evals = 0

        for block in range(prefill_blocks, num_blocks):
            block_start = block * block_length
            block_end = (block + 1) * block_length
            cur_x = x[:, block_start:block_end].clone()
            cur_attention = attention[:, block_start:block_end, :block_end]
            cur_position_ids = position_ids[:, block_start:block_end]

            global_positions = torch.arange(block_start, block_end, device=self.device)
            generated_mask = (
                (global_positions >= prompt_length)
                & (global_positions < prompt_length + gen_length)
            )
            generated_count = int(generated_mask.sum().item())
            local_acc = (
                TrajectoryAccumulator(prompt_length=0, mask_id=self.mask_id)
                if generated_count > 0
                else None
            )

            for step in range(denoising_steps + 1):
                mask_index = cur_x.eq(self.mask_id)
                if mask_index.sum() == 0:
                    self.model(
                        cur_x,
                        attention_mask=cur_attention,
                        position_ids=cur_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        store_kv=True,
                    )
                    forward_evals += 1
                    break

                logits = self.model(
                    cur_x,
                    attention_mask=cur_attention,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=False,
                ).logits
                forward_evals += 1

                if local_acc is not None:
                    local_acc.observe(
                        step=step,
                        logits=logits[:, generated_mask, :],
                        state_ids=cur_x[:, generated_mask],
                    )

                proposal, proposal_prob = self._sample(
                    logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )

                if step >= len(counts):
                    # The upstream loop permits an extra observation/store pass.
                    break

                transfer = torch.zeros_like(proposal, dtype=torch.bool)
                if remasking_strategy == "sequential":
                    for j in range(cur_x.shape[0]):
                        masked = mask_index[j].nonzero(as_tuple=True)[0]
                        if len(masked):
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
                        if k > 0:
                            idx = torch.topk(confidence[j], k=k).indices
                            transfer[j, idx] = True

                cur_x[transfer] = proposal[transfer]

            x[:, block_start:block_end] = cur_x
            if local_acc is not None:
                features, summaries = local_acc.finalize()
                block_records.append((generated_count, features, summaries))

            generated_segment = x[:, prompt_length : prompt_length + gen_length]
            if self.stop_ids and any(bool(generated_segment.eq(stop).any()) for stop in self.stop_ids):
                break

        generated_ids = x[0, prompt_length : prompt_length + gen_length]
        text = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True).strip()
        features, summaries = _combine_block_features(block_records)

        return GenerationResult(
            text=text,
            token_ids=[int(v) for v in generated_ids.tolist()],
            features=features,
            steps=summaries,
            metadata={
                "model_id": self.model_id,
                "model_revision": self.revision,
                "mask_id": self.mask_id,
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
                "telemetry_scope": "generated positions within each active SDAR block",
                "trajectory_semantics": "active_unresolved_positions_v4",
                "gen_length": int(gen_length),
                "block_length": int(block_length),
                "denoising_steps": int(denoising_steps),
                "temperature": float(temperature),
                "top_k": int(top_k),
                "top_p": float(top_p),
                "remasking_strategy": remasking_strategy,
                "confidence_threshold": float(confidence_threshold),
                "model_forward_evaluations": int(forward_evals),
                "parity_status": "PENDING_GPU_AUDIT",
            },
        )
