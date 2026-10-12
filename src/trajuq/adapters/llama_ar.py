from __future__ import annotations

import math

import numpy as np

from trajuq.adapters.base import BaseAdapter
from trajuq.schema import GenerationResult, StepSummary


TELEMETRY_SCHEMA_VERSION = "trajuq-ar-token-trace-v1"


def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    x = np.linspace(0.0, 1.0, values.size)
    x -= x.mean()
    denominator = float(np.dot(x, x))
    return 0.0 if denominator == 0.0 else float(np.dot(x, values - values.mean()) / denominator)


def summarize_ar_trace(
    entropy: list[float],
    confidence: list[float],
    margin: list[float],
    selected_nll: list[float],
) -> dict[str, float]:
    """Compute the prospectively frozen shared AR uncertainty summaries."""

    ent = np.asarray(entropy, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    mar = np.asarray(margin, dtype=float)
    nll = np.asarray(selected_nll, dtype=float)
    if not ent.size or not (ent.size == conf.size == mar.size == nll.size):
        raise ValueError("AR token trace arrays must be non-empty and aligned.")
    late_start = max(0, int(math.floor(0.75 * ent.size)))
    return {
        "trajectory_steps": float(ent.size),
        "mean_token_entropy": float(ent.mean()),
        "max_token_entropy": float(ent.max()),
        "late_token_entropy": float(ent[late_start:].mean()),
        "mean_selected_token_nll": float(nll.mean()),
        "min_token_margin": float(mar.min()),
        "entropy_slope": _slope(ent),
        "entropy_volatility": float(ent.std(ddof=0)),
        "entropy_mean": float(ent.mean()),
        "entropy_p90": float(np.quantile(ent, 0.90)),
        "entropy_max": float(ent.max()),
        "entropy_final": float(ent[-1]),
        "confidence_mean": float(conf.mean()),
        "confidence_final": float(conf[-1]),
        "margin_mean": float(mar.mean()),
        "margin_min": float(mar.min()),
        "margin_final": float(mar[-1]),
    }


class LlamaARAdapter(BaseAdapter):
    """Greedy Llama 3.1 adapter with pre-token-commit uncertainty telemetry."""

    name = "llama_ar"

    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        revision: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        if revision is None:
            raise ValueError("The prospective RQ4 model requires an exact revision.")

        self.torch = torch
        self.device = device
        self.model_id = model_id
        self.revision = revision
        dtype_obj = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype_obj,
        ).to(device).eval()

    def generate(
        self,
        prompt: str,
        seed: int = 2027,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        top_k_telemetry: int = 20,
        **_: object,
    ) -> GenerationResult:
        if do_sample:
            raise ValueError("RQ4 primary telemetry is frozen to deterministic greedy decoding.")
        if top_k_telemetry < 2:
            raise ValueError("top_k_telemetry must be at least 2.")

        torch = self.torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )

        generated_ids = output.sequences[0, input_ids.shape[1] :]
        if len(output.scores) != int(generated_ids.numel()):
            raise RuntimeError("Generated-token and pre-commit score counts do not align.")

        entropy: list[float] = []
        confidence: list[float] = []
        margin: list[float] = []
        selected_nll: list[float] = []
        token_trace: list[dict[str, object]] = []
        summaries: list[StepSummary] = []

        for step, (scores, chosen) in enumerate(zip(output.scores, generated_ids)):
            logp = torch.log_softmax(scores[0].float(), dim=-1)
            probabilities = logp.exp()
            normalized_entropy = float(
                (-(probabilities * logp).sum() / max(math.log(scores.shape[-1]), 1.0)).item()
            )
            top_values, top_ids = torch.topk(logp, k=min(top_k_telemetry, scores.shape[-1]))
            top_probabilities = top_values.exp()
            token_confidence = float(top_probabilities[0].item())
            token_margin = float((top_probabilities[0] - top_probabilities[1]).item())
            token_nll = float((-logp[int(chosen)]).item())

            entropy.append(normalized_entropy)
            confidence.append(token_confidence)
            margin.append(token_margin)
            selected_nll.append(token_nll)
            summaries.append(
                StepSummary(
                    step=step,
                    entropy_mean=normalized_entropy,
                    entropy_p90=normalized_entropy,
                    confidence_mean=token_confidence,
                    margin_mean=token_margin,
                    mask_fraction=0.0,
                    revision_fraction=0.0,
                    js_mean=0.0,
                    active_fraction=1.0,
                    active_count=1,
                )
            )
            token_trace.append(
                {
                    "step": step,
                    "selected_token_id": int(chosen),
                    "selected_token_logprob": -token_nll,
                    "entropy": normalized_entropy,
                    "margin": token_margin,
                    "top_token_ids": [int(value) for value in top_ids.tolist()],
                    "top_token_logprobs": [float(value) for value in top_values.tolist()],
                }
            )

        features = summarize_ar_trace(entropy, confidence, margin, selected_nll)
        text = self.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True).strip()
        return GenerationResult(
            text=text,
            token_ids=[int(token) for token in generated_ids.tolist()],
            features=features,
            steps=summaries,
            metadata={
                "model_id": self.model_id,
                "model_revision": self.revision,
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,
                "prompt_rendering": "apply_chat_template_single_user_add_generation_prompt",
                "telemetry_observation": "processed_prediction_scores_before_token_commit",
                "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
                "top_k_telemetry": int(top_k_telemetry),
                "token_trace": token_trace,
            },
        )
