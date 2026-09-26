from __future__ import annotations

from trajuq.adapters.base import BaseAdapter
from trajuq.schema import GenerationResult
from trajuq.trajectory import TrajectoryAccumulator


class LLaDAAdapter(BaseAdapter):
    """Compact instrumented low-confidence masked-diffusion decoder.

    Validate token-level parity against the chosen upstream checkpoint/revision
    before using this adapter for primary paper results.
    """

    name = "llada"

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        mask_id: int | None = None,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")

        self.torch = torch
        self.device = device
        self.model_id = model_id
        dtype_obj = getattr(torch, dtype)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype_obj,
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        inferred = self.tokenizer.mask_token_id
        if inferred is None and getattr(self.tokenizer, "mask_token", None):
            ids = self.tokenizer(
                self.tokenizer.mask_token,
                add_special_tokens=False,
            )["input_ids"]
            inferred = ids[0] if ids else None
        if mask_id is None:
            mask_id = inferred
        if mask_id is None and "LLaDA" in model_id:
            # Upstream LLaDA-8B generation code uses this id. It is a fallback,
            # not a substitute for freezing/checking the checkpoint tokenizer.
            mask_id = 126336
        if mask_id is None:
            raise ValueError("Could not infer mask token id; pass mask_id explicitly.")
        self.mask_id = int(mask_id)

    @staticmethod
    def _transfer_counts(torch, num_masked: int, steps: int, device):
        base = num_masked // steps
        rem = num_masked % steps
        counts = torch.full((steps,), base, dtype=torch.long, device=device)
        if rem:
            counts[:rem] += 1
        return counts

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

        if remasking != "low_confidence":
            raise ValueError("Schema-v1 adapter supports remasking='low_confidence' only.")
        if gen_length % block_length != 0:
            raise ValueError("gen_length must be divisible by block_length")
        num_blocks = gen_length // block_length
        if steps % num_blocks != 0:
            raise ValueError("steps must be divisible by the number of blocks")
        steps_per_block = steps // num_blocks

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt_ids = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)
        prompt_len = prompt_ids.shape[1]

        x = torch.full(
            (1, prompt_len + gen_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device,
        )
        x[:, :prompt_len] = prompt_ids
        prompt_index = x.ne(self.mask_id)

        acc = TrajectoryAccumulator(prompt_length=prompt_len, mask_id=self.mask_id)
        global_step = 0

        with torch.no_grad():
            for block in range(num_blocks):
                end = prompt_len + (block + 1) * block_length
                counts = self._transfer_counts(
                    torch,
                    block_length,
                    steps_per_block,
                    self.device,
                )

                for local_step in range(steps_per_block):
                    mask_index = x.eq(self.mask_id)

                    if cfg_scale > 0:
                        uncond = x.clone()
                        uncond[prompt_index] = self.mask_id
                        joined = torch.cat([x, uncond], dim=0)
                        joined_logits = self.model(joined).logits
                        cond_logits, uncond_logits = torch.chunk(joined_logits, 2, dim=0)
                        logits = uncond_logits + (cfg_scale + 1.0) * (
                            cond_logits - uncond_logits
                        )
                    else:
                        logits = self.model(x).logits

                    acc.observe(step=global_step, logits=logits, state_ids=x)
                    global_step += 1

                    if temperature > 0:
                        sample_probs = F.softmax(logits.float() / temperature, dim=-1)
                        proposal = torch.multinomial(
                            sample_probs.view(-1, sample_probs.shape[-1]),
                            num_samples=1,
                        ).view(sample_probs.shape[:-1])
                    else:
                        proposal = logits.argmax(dim=-1)

                    probs = F.softmax(logits.float(), dim=-1)
                    chosen_prob = probs.gather(-1, proposal.unsqueeze(-1)).squeeze(-1)
                    chosen_prob[:, end:] = -torch.inf
                    confidence = torch.where(mask_index, chosen_prob, -torch.inf)
                    transfer = torch.zeros_like(mask_index)

                    k = int(counts[local_step].item())
                    if k > 0:
                        idx = torch.topk(confidence[0], k=k).indices
                        transfer[0, idx] = True
                    x[transfer] = proposal[transfer]

        generated_ids = x[0, prompt_len:]
        text = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True).strip()
        features, summaries = acc.finalize()

        return GenerationResult(
            text=text,
            token_ids=[int(v) for v in generated_ids.tolist()],
            features=features,
            steps=summaries,
            metadata={
                "model_id": self.model_id,
                "mask_id": self.mask_id,
                "steps_requested": int(steps),
                "gen_length": int(gen_length),
                "block_length": int(block_length),
                "temperature": float(temperature),
                "cfg_scale": float(cfg_scale),
                "remasking": remasking,
            },
        )
