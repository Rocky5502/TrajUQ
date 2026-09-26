from __future__ import annotations

from trajuq.adapters.base import BaseAdapter
from trajuq.schema import GenerationResult
from trajuq.trajectory import TrajectoryAccumulator


UPSTREAM_REPOSITORY = "ML-GSAI/LLaDA"
UPSTREAM_COMMIT = "9182493720ed723ef8031210d85959364e51cbe0"


class LLaDAAdapter(BaseAdapter):
    """Instrumented masked-diffusion decoder following the official LLaDA sampler.

    The token-update semantics mirror ML-GSAI/LLaDA's public generate.py at the
    pinned integration commit for the default low-confidence setting. TrajUQ
    telemetry is inserted after logits are produced and before sampling/transfer.
    Primary-paper use remains gated on exact token-level parity at the frozen
    checkpoint revision.
    """

    name = "llada"

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        mask_id: int | None = None,
        revision: str | None = None,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

        self.torch = torch
        self.device = device
        self.model_id = model_id
        self.revision = revision
        dtype_obj = getattr(torch, dtype)

        self.model = AutoModel.from_pretrained(
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
        if mask_id is None and "llada" in model_id.lower():
            # Historical LLaDA-8B fallback from the official sampler. The
            # preflight/parity gate must still verify the selected checkpoint.
            mask_id = 126336
        if mask_id is None:
            raise ValueError("Could not infer mask token id; pass mask_id explicitly.")
        self.mask_id = int(mask_id)

    @staticmethod
    def _transfer_counts(torch, mask_index, steps: int):
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

    @staticmethod
    def _add_gumbel_noise(torch, logits, temperature: float):
        """Faithful to the public LLaDA Gumbel-max helper."""
        if temperature == 0:
            return logits
        logits64 = logits.to(torch.float64)
        noise = torch.rand_like(logits64, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits64.exp() / gumbel_noise

    def generate(
        self,
        prompt: str,
        seed: int = 2027,
        gen_length: int = 128,
        steps: int = 128,
        block_length: int = 128,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        **_: object,
    ) -> GenerationResult:
        torch = self.torch
        import torch.nn.functional as F

        if remasking not in {"low_confidence", "random"}:
            raise ValueError("LLaDA remasking must be 'low_confidence' or 'random'.")
        if gen_length % block_length != 0:
            raise ValueError("gen_length must be divisible by block_length")
        num_blocks = gen_length // block_length
        if steps % num_blocks != 0:
            raise ValueError("steps must be divisible by the number of blocks")
        steps_per_block = steps // num_blocks

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids = encoded.input_ids.to(self.device)
        prompt_attention = encoded.attention_mask.to(self.device)
        prompt_len = prompt_ids.shape[1]

        x = torch.full(
            (1, prompt_len + gen_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device,
        )
        x[:, :prompt_len] = prompt_ids
        attention_mask = torch.cat(
            [
                prompt_attention,
                torch.ones(
                    (1, gen_length),
                    dtype=prompt_attention.dtype,
                    device=self.device,
                ),
            ],
            dim=-1,
        )
        prompt_index = x.ne(self.mask_id)

        acc = TrajectoryAccumulator(prompt_length=prompt_len, mask_id=self.mask_id)
        global_step = 0
        forward_evals = 0

        with torch.no_grad():
            for block in range(num_blocks):
                block_start = prompt_len + block * block_length
                block_end = prompt_len + (block + 1) * block_length
                block_mask_index = x[:, block_start:block_end].eq(self.mask_id)
                transfer_counts = self._transfer_counts(
                    torch,
                    block_mask_index,
                    steps_per_block,
                )

                for local_step in range(steps_per_block):
                    mask_index = x.eq(self.mask_id)

                    if cfg_scale > 0:
                        uncond = x.clone()
                        uncond[prompt_index] = self.mask_id
                        joined = torch.cat([x, uncond], dim=0)
                        joined_attention = torch.cat([attention_mask, attention_mask], dim=0)
                        joined_logits = self.model(
                            joined,
                            attention_mask=joined_attention,
                        ).logits
                        forward_evals += 1
                        cond_logits, uncond_logits = torch.chunk(joined_logits, 2, dim=0)
                        logits = uncond_logits + (cfg_scale + 1.0) * (
                            cond_logits - uncond_logits
                        )
                    else:
                        logits = self.model(
                            x,
                            attention_mask=attention_mask,
                        ).logits
                        forward_evals += 1

                    active_positions = torch.zeros_like(x, dtype=torch.bool)
                    active_positions[:, block_start:block_end] = True
                    acc.observe(
                        step=global_step,
                        logits=logits,
                        state_ids=x,
                        active_position_mask=active_positions,
                    )
                    global_step += 1

                    noisy_logits = self._add_gumbel_noise(
                        torch,
                        logits,
                        temperature=temperature,
                    )
                    proposal = torch.argmax(noisy_logits, dim=-1)

                    if remasking == "low_confidence":
                        probs = F.softmax(logits, dim=-1)
                        proposal_prob = probs.gather(
                            -1,
                            proposal.unsqueeze(-1),
                        ).squeeze(-1)
                    else:
                        proposal_prob = torch.rand(
                            proposal.shape,
                            device=self.device,
                        )

                    proposal_prob[:, block_end:] = -torch.inf
                    proposal = torch.where(mask_index, proposal, x)
                    confidence = torch.where(mask_index, proposal_prob, -torch.inf)
                    transfer = torch.zeros_like(proposal, dtype=torch.bool)

                    for j in range(confidence.shape[0]):
                        k = int(transfer_counts[j, local_step].item())
                        if k <= 0:
                            continue
                        idx = torch.topk(confidence[j], k=k).indices
                        transfer[j, idx] = True
                    x[transfer] = proposal[transfer]

        generated_ids = x[0, prompt_len:]
        text = self.tokenizer.decode(
            generated_ids.tolist(),
            skip_special_tokens=True,
        ).strip()
        features, summaries = acc.finalize()

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
                "steps_requested": int(steps),
                "model_forward_evaluations": int(forward_evals),
                "gen_length": int(gen_length),
                "block_length": int(block_length),
                "temperature": float(temperature),
                "cfg_scale": float(cfg_scale),
                "remasking": remasking,
                "trajectory_semantics": "active_unresolved_positions_v2",
                "parity_status": "PENDING_GPU_AUDIT",
            },
        )
