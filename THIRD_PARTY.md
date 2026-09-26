# Third-party research assets

TrajUQ code is MIT licensed. This repository does **not** relicense model weights, upstream runtimes, or datasets.

- **LLaDA-MoE** — selected model: `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`; the inspected model card reports Apache-2.0. The exact runtime/model revision must be frozen before primary use.
- **SDAR** — selected model: `JetLM/SDAR-1.7B-Chat`; the inspected model card reports Apache-2.0. Public repository `JetAstra/SDAR` was MIT at the inspected revision.
- **Dream** — selected model: `Dream-org/Dream-v0-Instruct-7B`; model/repository materials inspected for this setup report Apache-2.0.
- **FLARE / HybridDiffusion** — selected checkpoint: `yuchen-zhu-zyc/HybridDiffusion-2B`. The inspected upstream release states PolyForm Noncommercial 1.0.0 for HybridDiffusion-authored code/modifications/checkpoints; third-party portions retain their own terms. Do not describe this checkpoint as MIT/Apache or assume OSI-open-source rights.
- **Benchmarks** — GSM8K, MATH-500, GPQA, ARC, MMLU-Pro, and TruthfulQA are downloaded from upstream sources. Their cards/licenses/terms control redistribution and use.

Always re-check exact upstream terms and revisions before public artifact release.


## SDAR adapter attribution

`src/trajuq/adapters/sdar.py` follows and modifies the block-diffusion inference structure published in `JetAstra/SDAR` at commit `4c2749ba103448f45520e8411533710a1e66574d`. The inspected upstream repository is MIT licensed. TrajUQ adds telemetry, revision pinning, deterministic-greedy support for research controls, and its own output schema. The selected checkpoint `JetLM/SDAR-1.7B-Chat` is governed by its model-card terms (Apache-2.0 at the inspected revision).


## LLaDA adapter attribution

`src/trajuq/adapters/llada.py` follows the public LLaDA masked-diffusion generation algorithm in `ML-GSAI/LLaDA` at commit `9182493720ed723ef8031210d85959364e51cbe0`, with TrajUQ telemetry added before sampling/token transfer. The selected MoE checkpoint is `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`; its inspected model card reports Apache-2.0. Repository-code and checkpoint terms should be rechecked at final artifact release.
