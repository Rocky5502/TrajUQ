<div align="center">

# TrajUQ

### Trajectory-aware uncertainty across diffusion, autoregressive, and black-box language generation

<p>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-2ea44f">
  <img alt="Frozen protocol" src="https://img.shields.io/badge/Protocol-Frozen%20%26%20auditable-6f42c1">
  <img alt="Research artifact" src="https://img.shields.io/badge/Artifact-Reproducible-0a7ea4">
</p>

**Uncertainty is not only a final confidence score. TrajUQ studies how uncertainty evolves while a language model generates.**

</div>

---

TrajUQ is a research framework for measuring, calibrating, and using uncertainty across **diffusion language models**, an **autoregressive control**, and **black-box hosted LLMs**. It combines outcome-isolated telemetry, exact revision pinning, parity gates, calibrated selective prediction, adaptive compute, behavioral uncertainty, and evidence-bound statistical analysis.

> **Repository boundary:** this repository contains research code, frozen scientific configurations, manifests, tests, and reproducibility material. Manuscript/venue authoring files are intentionally maintained outside the repository.

> **Final scientific closure:** RQ1--RQ5 are complete and evidence-bound. The
> RQ5 primary comparison is retained as `NOT_SUPPORTED`; cross-model
> heterogeneity is `DESCRIPTIVE_ONLY`, and unmatched cross-observability
> comparisons remain `NOT_IDENTIFIABLE`. See
> [TRAJUQ_NOVELTY_DEFENSE.md](TRAJUQ_NOVELTY_DEFENSE.md),
> [CLOSEST_WORK_MATRIX.tsv](CLOSEST_WORK_MATRIX.tsv), and
> [EARLY_ACTIONABILITY_READINESS.md](EARLY_ACTIONABILITY_READINESS.md) for the
> claim boundary and secondary-prefix analysis contract.

## Contents

- [Reviewer quick path](#reviewer-quick-path)
- [Research questions](#research-questions)
- [LLM model matrix](#llm-model-matrix)
- [Datasets](#datasets)
- [How TrajUQ works](#how-trajuq-works)
- [Reproducibility levels](#reproducibility-levels)
- [Repository map](#repository-map)
- [Documentation](#documentation)
- [Security, licensing, and citation](#security-licensing-and-citation)

## Reviewer quick path

A reviewer can validate the released evidence layer **without a GPU, model download, or benchmark download**:

~~~bash
git clone https://github.com/Rocky5502/TrajUQ.git
cd TrajUQ

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-analysis.txt

python reproduction/reproduce_core.py
~~~

This verifies repository secret hygiene and the SHA-256-bound authorized artifact manifest.

For deeper CPU-side software verification:

~~~bash
pip install -e ".[dev,math]"
make verify
~~~

Start with **[docs/REVIEWER_GUIDE.md](docs/REVIEWER_GUIDE.md)** for the shortest evidence-focused tour.

## Research questions

| RQ | Question | Role |
|---|---|---|
| **RQ1 — Incremental predictive signal** | Do native denoising-trajectory features rank answer failures beyond calibrated decision-time uncertainty? | Primary diffusion study |
| **RQ2 — Selective generation** | Does trajectory-aware risk improve held-out error ranking and risk–coverage behavior relative to registered comparators, including repeated sampling? | Selective-prediction study |
| **RQ3 — Adaptive compute** | Can calibrated risk allocate a second generation budget more effectively than fixed or matched-random refinement? | Compute-allocation study |
| **RQ4 — Autoregressive control** | Which temporal uncertainty dynamics are shared across diffusion and autoregressive generation, which are architecture-specific, and how well do risk rankings transfer? | Cross-paradigm external validity |
| **RQ5 — Black-box uncertainty** | How much usable uncertainty remains when only deployment-visible behavioral signals are available? | Black-box external validity |

Full definitions and claim boundaries: **[docs/RESEARCH_QUESTIONS.md](docs/RESEARCH_QUESTIONS.md)**.

## LLM model matrix

TrajUQ deliberately spans three observability regimes: native diffusion trajectories, autoregressive token trajectories, and behavioral black-box uncertainty.

| Model | Paradigm | Access | Study role | Parameters | Frozen revision / provider-visible identifier | Uncertainty view |
|---|---|---|---|---:|---|---|
| [LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) | Masked diffusion MoE | Open-weight | RQ1–RQ3 primary | 7.36B total / 1.4B active | \`783d3467f108d28ac0a78d3e41af16ab05cabd8d\` | entropy, margin, drift, revisions, persistence, commitment |
| [SDAR-1.7B-Chat](https://huggingface.co/JetLM/SDAR-1.7B-Chat) | Blockwise diffusion | Open-weight | RQ1–RQ3 primary | 2.03B | \`14fcc4b796f05b9a11aa565b2353199af757d079\` | blockwise trajectory uncertainty + revisions |
| [Dream-v0-Instruct-7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) | Masked diffusion | Open-weight | RQ1–RQ3 primary | 7.62B | \`05334cb9faaf763692dcf9d8737c642be2b2a6ae\` | pre-transfer trajectory uncertainty |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Autoregressive | Open-weight | RQ4 control | 8.03B | \`0e9e39f249a16976918f6564b8830bc894c89659\` | token entropy, margin, selected-token log probability |
| **DeepSeek-V3.2** | Hosted LLM | Black-box API | RQ5 | provider-managed / not assumed observable | \`Pro/deepseek-ai/DeepSeek-V3.2\` | repeated-sample disagreement, consistency, verbalized confidence |
| **GLM-5.1** | Hosted LLM | Black-box API | RQ5 | provider-managed / not assumed observable | \`Pro/zai-org/GLM-5.1\` | repeated-sample disagreement, consistency, verbalized confidence |
| **Kimi-K2.6** | Hosted LLM | Black-box API | RQ5 | provider-managed / not assumed observable | \`Pro/moonshotai/Kimi-K2.6\` | repeated-sample disagreement, consistency, verbalized confidence |
| [HybridDiffusion-2B](https://huggingface.co/yuchen-zhu-zyc/HybridDiffusion-2B) | Hybrid AR/diffusion | Open-weight diagnostic | Diagnostic only | upstream 2B-class checkpoint | upstream runtime pinned separately | same-checkpoint AR/diffusion comparison |

For hosted RQ5 models, provider-side weights, exact backend revisions, routing, precision, and hardware are **not treated as observable facts**. The provider-visible model IDs above are the reproducibility boundary.

Exact configurations, parameter evidence, prompts, decoding settings, and telemetry contracts: **[docs/MODELS.md](docs/MODELS.md)** and **[manifests/models.json](manifests/models.json)**.

## Datasets

| Dataset | Source | Evaluation policy |
|---|---|---|
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | official train → calibration; official test → evaluation; deterministic caps |
| MATH-500 | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | deterministic internal holdout of the source test split |
| GPQA-Diamond | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) | gated source; deterministic internal holdout |
| ARC-Challenge | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | official validation → calibration; official test → evaluation |
| MMLU-Pro | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | official validation → calibration; official test → evaluation |
| TruthfulQA | [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa) | deterministic internal holdout of validation |

Exact source IDs, immutable revisions, split rules, caps, scorer boundaries, and redistribution policy: **[docs/DATASETS.md](docs/DATASETS.md)**.

## How TrajUQ works

~~~mermaid
flowchart LR
    A[Generation] --> B[Outcome-isolated uncertainty telemetry]
    B --> C[Immutable raw artifact]
    C --> D[Schema + provenance audit]
    D --> E[Separate correctness scoring]
    E --> F[Calibration only on calibration partition]
    F --> G[Held-out evaluation]
    G --> H[Registered statistics + claim gate]
    H --> I[Authorized evidence artifact]

    J[Exact model/data revisions] --> A
    K[Parity gates] --> A
    L[Security + hash manifests] --> I
~~~

The key rule is:

~~~text
artifact → audit → statistic → evidence gate → authorized scientific claim
~~~

Operational progress, GPU utilization, partial outputs, or implemented code do not substitute for completed evidence.

## Reproducibility levels

### Level 1 — Evidence verification

No GPU or benchmark download:

~~~bash
python reproduction/reproduce_core.py
~~~

### Level 2 — CPU software verification

Runs security, lint, unit tests, and smoke checks:

~~~bash
pip install -e ".[dev,math]"
make verify
~~~

### Level 3 — Full experimental reproduction

Requires model and dataset access plus the registered compute environment:

- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Model contracts](docs/MODELS.md)
- [Dataset contracts](docs/DATASETS.md)
- [Data split policy](docs/DATA_SPLITS.md)
- [A100 execution](docs/A100_EXECUTION.md)
- [Evidence map](docs/EVIDENCE_MAP.md)

The authoritative frozen experiment implementation is:

\`28002055c828c6910840de6f2105aa6367514144\`

Later repository commits may improve documentation, security, packaging, and usability without rewriting that experiment identity.

## Repository map

~~~text
TrajUQ/
├── configs/          frozen experiment, model, schema, and operations contracts
├── data/             dataset source manifests + synthetic fixtures
├── docs/             reviewer, protocol, reproduction, model, and data documentation
├── manifests/        concise model, dataset, and environment provenance
├── reproduction/     lightweight evidence-verification entry point
├── results/          small authorized evidence artifacts only
├── scripts/          generation, audits, statistics, security, export, packaging
├── src/trajuq/       reusable TrajUQ library
└── tests/            protocol, integrity, parity, statistics, and regression tests
~~~

Raw benchmark rows, gated content, model weights, caches, credentials, and manuscript source are intentionally excluded.

## Documentation

| Goal | Start here |
|---|---|
| Review the artifact quickly | [Reviewer guide](docs/REVIEWER_GUIDE.md) |
| Understand all RQs | [Research questions](docs/RESEARCH_QUESTIONS.md) |
| Inspect LLM settings | [Models](docs/MODELS.md) |
| Inspect dataset provenance | [Datasets](docs/DATASETS.md) |
| Trace evidence | [Evidence map](docs/EVIDENCE_MAP.md) |
| Reproduce experiments | [Reproducibility](docs/REPRODUCIBILITY.md) |
| Run on A100 | [A100 execution](docs/A100_EXECUTION.md) |
| Fix common problems | [Troubleshooting](docs/TROUBLESHOOTING.md) |
| Browse all docs | [Documentation index](docs/README.md) |

## Security, licensing, and citation

- **Security:** [SECURITY.md](SECURITY.md)
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Third-party terms:** [THIRD_PARTY.md](THIRD_PARTY.md)
- **License:** [MIT](LICENSE)
- **Software citation:** [CITATION.cff](CITATION.cff)

TrajUQ source code is MIT licensed. Model weights, upstream runtimes, datasets, and hosted provider services retain their own terms.

---

<div align="center">

**Built for inspectable uncertainty research: frozen protocols, explicit boundaries, and evidence before claims.**

</div>
