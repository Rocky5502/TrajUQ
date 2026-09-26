# Upstream integration map

Pinned commits below are **integration reference points**, not vendored code.

| Family | Public checkpoint | Repository | Reference commit | Integration plan |
|---|---|---|---|---|
| LLaDA-MoE | `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` | LLaDA public implementations | freeze checkpoint/runtime revision before experiment | pilot masked-diffusion adapter; parity-audit the selected checkpoint |
| SDAR | `JetLM/SDAR-1.7B-Chat` | `JetAstra/SDAR` | `4c2749ba103448f45520e8411533710a1e66574d` | instrument upstream block-diffusion loop at pre-transfer logits and post-step state |
| Dream | `Dream-org/Dream-v0-Instruct-7B` | `DreamLM/Dream` | `31f94a60d187e3fd481fee3bbc2c732eb94a879c` | use the documented generation hook |
| FLARE / HybridDiffusion | `yuchen-zhu-zyc/HybridDiffusion-2B` | `yuchen-zhu-zyc/HybridDiffusion` | `6ca547aebb72bfe897e80e0a683c776789e5f38c` | keep specialized runtime external; add compact diffusion telemetry and separate AR diagnostic |

Do not silently update upstream revisions during a paper experiment. Model checkpoint terms can differ from repository-code licenses.
