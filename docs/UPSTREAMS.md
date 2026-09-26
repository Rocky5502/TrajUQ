# Upstream integration map

Pinned commits are integration reference points, not vendored code.

| Family | Repository | Reference commit | Integration plan |
|---|---|---|---|
| LLaDA | lrbob/LLaDA | 38c5a6c6718cf286b6317baa30466a1f4e3f9e77 | Pilot decoder is instrumented locally; validate parity before primary runs. |
| SDAR | JetAstra/SDAR | 4c2749ba103448f45520e8411533710a1e66574d | Instrument upstream block-diffusion loop at pre-transfer logits and post-step state. |
| FLARE | yuchen-zhu-zyc/HybridDiffusion | 6ca547aebb72bfe897e80e0a683c776789e5f38c | Keep external; add compact telemetry to diffusion path and use AR path only as a diagnostic. |
| Dream | DreamLM/Dream | 31f94a60d187e3fd481fee3bbc2c732eb94a879c | Use the public generation hook to observe state/logits without copying its generator. |

Clone runtimes under gitignored \`third_party/\` or elsewhere, checkout the pinned commit, install per upstream instructions, and record the actual commit in each run manifest. Do not silently update upstream code during the paper experiment.
