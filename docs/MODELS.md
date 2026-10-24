# Models and generation contracts

The experiment configuration files are the authoritative machine-readable contracts. This document provides a human-readable inventory.

> Some frozen configuration files retain historical status fields from the point at which the protocol was frozen. Those fields are provenance, not a live campaign dashboard.

## Primary diffusion models

### LLaDA-MoE-7B-A1B-Instruct

- Source: https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct
- Revision: 783d3467f108d28ac0a78d3e41af16ab05cabd8d
- Architecture: masked diffusion mixture-of-experts
- Parameters: 7,356,880,896 total; 1.4B active reported by the official model card
- Precision/device: bfloat16 / CUDA
- Generation: length 128, 128 steps, block length 128, temperature 0.0, CFG 0.0, low-confidence remasking
- Config: configs/models/llada.yaml
- Upstream generation reference: ML-GSAI/LLaDA at 9182493720ed723ef8031210d85959364e51cbe0

### SDAR-1.7B-Chat

- Source: https://huggingface.co/JetLM/SDAR-1.7B-Chat
- Revision: 14fcc4b796f05b9a11aa565b2353199af757d079
- Architecture: blockwise diffusion language model
- Parameters: 2,031,739,904
- Precision/device: bfloat16 / CUDA
- Native generation: length 128, block length 4, 4 denoising steps, temperature 1.0, top-k 0, top-p 1.0, low-confidence dynamic remasking, confidence threshold 0.85
- Native stochastic parity: seed 2027, 20 prompts, exact token-ID equality
- Config: configs/models/sdar.yaml
- Upstream runtime reference: JetAstra/SDAR at 4c2749ba103448f45520e8411533710a1e66574d

### Dream-v0-Instruct-7B

- Source: https://huggingface.co/Dream-org/Dream-v0-Instruct-7B
- Revision: 05334cb9faaf763692dcf9d8737c642be2b2a6ae
- Architecture: masked diffusion language model
- Parameters: 7,615,616,512
- Precision/device: bfloat16 / CUDA
- Generation: max 128 new tokens, 128 steps, temperature 0.0, top-p 1.0, entropy transfer, algorithm temperature 0.1
- Telemetry: documented pre-transfer logits-hook contract
- Config: configs/models/dream.yaml

## RQ4 autoregressive control

### Llama-3.1-8B-Instruct

- Source: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- Revision: 0e9e39f249a16976918f6564b8830bc894c89659
- Architecture: autoregressive decoder-only transformer
- Parameters: 8,030,261,248
- Precision/device: bfloat16 / CUDA
- Generation: seed 2027, deterministic greedy decoding, max 512 new tokens
- Telemetry: pre-token-commit entropy, margin, selected-token log probability, top-20 token information; full logits are not stored
- Config: configs/models/llama31_ar.yaml

This control is a prospective extension and is not counted as a fourth diffusion family in the original RQ1 gate.

## RQ5 black-box models

Provider: SiliconFlow chat-completions API.

Frozen provider-visible identifiers:
- Pro/deepseek-ai/DeepSeek-V3.2
- Pro/zai-org/GLM-5.1
- Pro/moonshotai/Kimi-K2.6

Registered decoding: five repeated samples, temperature 0.6, top-p 0.95, max output tokens 256, thinking enabled, thinking budget 512, no provider seed assumed, non-streaming.

Provider-side revisions, parameter counts, routing, precision, and hardware are not treated as known.

Config: configs/experiments/tmlr_rq5_blackbox.yaml

## Diagnostic model

### HybridDiffusion-2B / FLARE

- Source: https://huggingface.co/yuchen-zhu-zyc/HybridDiffusion-2B
- Upstream runtime: yuchen-zhu-zyc/HybridDiffusion at 6ca547aebb72bfe897e80e0a683c776789e5f38c
- Role: same-checkpoint autoregressive/diffusion diagnostic only
- Precision/device: bfloat16 / CUDA
- License note: inspected HybridDiffusion-authored material uses PolyForm Noncommercial 1.0.0
- Config: configs/models/flare.yaml

It is not part of the primary confirmatory model matrix.

## Parameter evidence

Machine-readable parameter evidence is stored in configs/operations/model_parameter_registry.json. Open-weight counts are derived from official Hub safetensors metadata at the pinned revisions where available.
