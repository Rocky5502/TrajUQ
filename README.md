# TrajUQ

**Trajectory-aware uncertainty across diffusion, autoregressive, and black-box language generation**

TrajUQ studies whether uncertainty is more informative when it is measured across a model's generation process rather than only at the final decision. The repository implements leak-safe trajectory telemetry, calibrated error-risk estimation, selective prediction, adaptive compute, autoregressive controls, black-box behavioral uncertainty, provenance checks, and reproducible statistical analysis.

> **Research software repository.** Manuscript source and venue-specific authoring files are intentionally maintained outside this repository. This repository contains the implementation, frozen scientific configurations, manifests, tests, and reproducibility material.

## Scientific scope

TrajUQ is organized around five research questions:

| RQ | Question | Scientific role |
|---|---|---|
| **RQ1 — Incremental predictive signal** | Do native denoising-trajectory features rank answer failures beyond the same model's calibrated decision-time uncertainty? | Primary diffusion study |
| **RQ2 — Selective generation** | Does calibrated trajectory risk improve held-out error ranking and risk–coverage behavior relative to registered comparators, including repeated sampling? | Selective-prediction extension |
| **RQ3 — Adaptive compute** | Can frozen calibrated risk allocate a second generation budget more effectively than matched random or fixed refinement? | Compute-allocation study |
| **RQ4 — Autoregressive control** | Which temporal uncertainty dynamics are shared between diffusion and autoregressive generation, which are architecture-specific, and how well do calibrated risk rankings transfer? | Prospective external-validity control |
| **RQ5 — Black-box uncertainty** | How much useful uncertainty remains when only deployment-visible behavioral signals are available through hosted APIs? | Black-box external-validity study |

Detailed definitions and evidence boundaries are in docs/RESEARCH_QUESTIONS.md.

## Core design principles

- **Outcome isolation.** Raw trajectory features are produced before benchmark correctness is joined.
- **Frozen revisions.** Primary models, datasets, split rules, seeds, and statistical gates are pinned before confirmatory evaluation.
- **Parity before evidence.** Instrumented generation must match the relevant native sampler contract before its outputs are eligible.
- **Calibration/test separation.** Probability calibration and policy selection use calibration data only; final metrics use held-out evaluation data.
- **Fail-closed evidence.** Generated, saved, audited, scored, statistically complete, and claim-authorized are treated as different states.
- **Negative results are retained.** Model-family heterogeneity, null effects, and adverse calibration behavior are not filtered out.

The authoritative experiment implementation used for the frozen campaign is commit 28002055c828c6910840de6f2105aa6367514144. Later repository commits improve documentation, reproducibility, and release engineering without rewriting that experimental provenance.

## Models

| Role | Model | Exact revision | Precision | Main generation contract |
|---|---|---|---|---|
| Primary diffusion | [LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) | 783d3467f108d28ac0a78d3e41af16ab05cabd8d | bfloat16 | 128-token generation, 128 steps, low-confidence remasking |
| Primary diffusion | [SDAR-1.7B-Chat](https://huggingface.co/JetLM/SDAR-1.7B-Chat) | 14fcc4b796f05b9a11aa565b2353199af757d079 | bfloat16 | block length 4, 4 denoising steps, native stochastic parity contract |
| Primary diffusion | [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) | 05334cb9faaf763692dcf9d8737c642be2b2a6ae | bfloat16 | 128-token generation, 128 steps, entropy transfer |
| RQ4 AR control | [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 0e9e39f249a16976918f6564b8830bc894c89659 | bfloat16 | deterministic greedy trace, max 512 new tokens |
| Diagnostic only | [HybridDiffusion-2B](https://huggingface.co/yuchen-zhu-zyc/HybridDiffusion-2B) | upstream runtime pinned separately | bfloat16 | same-checkpoint AR/diffusion diagnostic; not a primary endpoint |

RQ5 uses provider-visible model identifiers only:
- Pro/deepseek-ai/DeepSeek-V3.2
- Pro/zai-org/GLM-5.1
- Pro/moonshotai/Kimi-K2.6

The complete model contracts, parameter evidence, generation settings, and limitations are documented in docs/MODELS.md and manifests/models.json.

## Datasets

TrajUQ does **not** redistribute benchmark rows. The repository records source identifiers, immutable revisions, deterministic split policies, and hashes.

| Dataset | Upstream source | Evaluation policy |
|---|---|---|
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | official train → calibration; official test → evaluation; pre-generation caps |
| MATH-500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | deterministic internal holdout from the source test split |
| GPQA-Diamond | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) | gated source; deterministic internal holdout |
| ARC-Challenge | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | official validation → calibration; test → evaluation |
| MMLU-Pro | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | official validation → calibration; test → evaluation |
| TruthfulQA | [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa) | deterministic internal holdout of validation |

Exact revisions, configs, split sizes, cap policy, and redistribution boundaries are in docs/DATASETS.md, data/manifests/evaluation_splits.json, and manifests/datasets.json.

## Environment

The authoritative open-weight campaign was attested on Ubuntu 24.04.3 LTS, Python 3.12.3, PyTorch 2.6.0 + CUDA 12.4, Transformers 4.52.4, one NVIDIA A100-SXM4-80GB, and bfloat16 inference.

See manifests/environment.json and requirements-experiment.txt.

## Installation

### CPU analysis / artifact validation

~~~bash
git clone https://github.com/Rocky5502/TrajUQ.git
cd TrajUQ
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-analysis.txt
python reproduction/reproduce_core.py
~~~

### Development

~~~bash
pip install -e ".[dev,math]"
pytest -q
python scripts/security_scan.py --repo .
~~~

### Full model experiments

Install the correct CUDA-enabled PyTorch build first, then:

~~~bash
pip install -r requirements-experiment.txt
~~~

Full generation additionally requires access to the corresponding model checkpoints and benchmark sources. GPQA is gated.

## Repository layout

~~~text
TrajUQ/
├── configs/          # frozen experiment, model, schema, and operations contracts
├── data/             # source manifests and synthetic smoke data; raw benchmarks excluded
├── docs/             # protocols, RQs, model/data documentation, execution guides
├── manifests/        # concise model, dataset, and environment provenance
├── reproduction/     # CPU-safe artifact verification entry point
├── results/          # small authorized evidence only; large/transient outputs excluded
├── scripts/          # generation, audits, scoring, statistics, packaging, security
├── src/trajuq/       # reusable TrajUQ library
└── tests/            # protocol, parity, integrity, and analysis tests
~~~

## Reproducibility

The default reproduction path validates released authorized evidence without downloading models or datasets:

~~~bash
python reproduction/reproduce_core.py
~~~

For full experimental reproduction, start with docs/EXPERIMENT_PROTOCOL.md, docs/REPRODUCIBILITY.md, docs/DATA_SPLITS.md, and docs/A100_EXECUTION.md.

## Security

Credentials, API keys, rental connection files, private keys, model caches, and raw gated data must never be committed. CI runs a repository secret-hygiene scan, Dependabot monitors Python and GitHub Actions dependencies, and release tooling verifies repository identity before packaging or guarded pushes.

See SECURITY.md.

## Licensing

Original TrajUQ source code is released under the [MIT License](LICENSE). Model weights, upstream runtimes, datasets, and provider services retain their own licenses and terms. See THIRD_PARTY.md and THIRD_PARTY_NOTICES.md.

## Citation

Software citation metadata are provided in CITATION.cff. A paper citation should be added only after the associated manuscript is public.
