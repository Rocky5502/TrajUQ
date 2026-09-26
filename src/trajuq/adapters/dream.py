from __future__ import annotations

from trajuq.adapters.base import BaseAdapter
from trajuq.schema import GenerationResult
from trajuq.trajectory import TrajectoryAccumulator


class DreamAdapter(BaseAdapter):
    """Adapter for Dream checkpoints exposing generation_tokens_hook_func."""

    name = "dream"

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
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
            torch_dtype=dtype_obj,
            trust_remote_code=True,
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
        )

    def generate(
        self,
        prompt: str,
        seed: int = 2027,
        max_new_tokens: int = 128,
        steps: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        alg: str = "entropy",
        alg_temp: float = 0.1,
        **_: object,
    ) -> GenerationResult:
        torch = self.torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        acc = TrajectoryAccumulator(
            prompt_length=input_ids.shape[1],
            mask_id=self.tokenizer.mask_token_id,
        )

        def hook(step, x, logits):
            acc.observe(step=step, logits=logits, state_ids=x)
            return x

        with torch.no_grad():
            output = self.model.diffusion_generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=alg_temp,
                generation_tokens_hook_func=hook,
            )

        generated_ids = output.sequences[0][input_ids.shape[1] :]
        text = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True).strip()
        features, summaries = acc.finalize()

        return GenerationResult(
            text=text,
            token_ids=[int(x) for x in generated_ids.tolist()],
            features=features,
            steps=summaries,
            metadata={
                "model_id": self.model_id,
                "model_revision": self.revision,
                "steps_requested": int(steps),
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "algorithm": alg,
                "algorithm_temperature": float(alg_temp),
                "trajectory_semantics": "active_unresolved_positions_v2",
            },
        )
