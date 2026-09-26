# TrajUQ

**Trajectory-Aware Uncertainty for Diffusion Language Models**

TrajUQ studies whether the *denoising trajectory itself* can predict when a diffusion language model is likely to be wrong, and whether that uncertainty can support calibrated selective generation and adaptive inference-time compute.

> **Status (2026-09-26): research scaffold + pilot implementation.**  
> The repository contains frozen data/model manifests, trajectory feature extraction, calibration/evaluation utilities, Dream and LLaDA pilot adapters, dataset download/normalization scripts, smoke tests, and an experiment protocol. SDAR/FLARE integration remains an explicit upstream boundary until its telemetry hooks are parity-validated. No experimental results are claimed yet.

## Research questions

- **RQ1 — Predictive signal:** Which denoising-trajectory signals predict incorrect answers beyond final-step confidence?
- **RQ2 — Selective generation:** Does trajectory-aware uncertainty improve calibration and risk–coverage behavior across model families/tasks?
- **RQ3 — Adaptive compute:** Can calibrated risk allocate extra denoising/remasking steps more effectively than fixed schedules or raw-confidence heuristics?
- **RQ4 — Failure source:** How much risk is associated with denoiser ambiguity versus sampler instability, and when available, does AR/diffusion disagreement reveal failures missed by a single path?

## Core idea

For every denoising step, TrajUQ records compact pre-outcome signals: normalized entropy, top-1 confidence, top-2 margin, masked-token fraction, commitment timing, token revisions, late revisions, and trajectory trends. These are pooled into example-level features and calibrated against correctness labels. The calibrated error probability can drive three actions:

\`ACCEPT → MORE_COMPUTE → ABSTAIN/FALLBACK\`.

## Layout

\`\`\`text
TrajUQ/
├── configs/
│   ├── experiments/pilot.yaml
│   └── models/{dream,llada,sdar,flare}.yaml
├── data/
│   ├── manifests/benchmarks.json
│   ├── samples/synthetic_smoke.jsonl
│   └── README.md
├── docs/
│   ├── PROTOCOL.md
│   ├── REPRODUCIBILITY.md
│   └── UPSTREAMS.md
├── scripts/
│   ├── download_data.py
│   ├── prepare_benchmarks.py
│   ├── run_generation.py
│   └── smoke_metrics.py
├── src/trajuq/
│   ├── adapters/
│   ├── calibration.py
│   ├── decision.py
│   ├── evaluation.py
│   ├── io.py
│   ├── schema.py
│   └── trajectory.py
├── tests/
├── THIRD_PARTY.md
├── EXPERIMENT_FREEZE_CHECKLIST.md
├── pyproject.toml
└── requirements.txt
\`\`\`

## Model plan

| Model | Role | Current repo status |
|---|---|---|
| FLARE-2B | same-checkpoint AR/diffusion diagnostic + hybrid dLM | upstream boundary; telemetry hook next |
| SDAR-1.7B | block-diffusion family | upstream boundary; telemetry hook next |
| LLaDA / LLaDA-MoE | masked diffusion family | pilot adapter implemented |
| Dream-v0-Instruct-7B | optional validation/extension family | hook-based adapter implemented |

The exact checkpoint revision used in experiments must be frozen in the run manifest.

## Dataset plan

Primary manifest: GSM8K, MATH-500, GPQA-Diamond, ARC-Challenge, MMLU-Pro, and TruthfulQA.

Raw benchmark copies are **not committed**. Dataset licenses/terms differ; GPQA is gated and asks users not to expose examples publicly; and raw snapshots without revision metadata are not reproducible. TrajUQ instead commits a machine-readable manifest plus deterministic download/normalization scripts. Local run metadata records dataset fingerprints/revisions.

## Quick start

\`\`\`bash
git clone https://github.com/Rocky5502/TrajUQ.git
cd TrajUQ
python -m venv .venv

# Windows:
# .venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e ".[dev]"
pytest -q
python scripts/smoke_metrics.py
\`\`\`

GPU/data setup:

\`\`\`bash
pip install -e ".[models]"
python scripts/download_data.py --dataset gsm8k math500 arc_challenge mmlu_pro truthfulqa
python scripts/prepare_benchmarks.py --dataset gsm8k --limit 50

python scripts/run_generation.py \
  --adapter llada \
  --model GSAI-ML/LLaDA-8B-Instruct \
  --input data/processed/gsm8k.jsonl \
  --output runs/pilot_llada_gsm8k.jsonl \
  --limit 50
\`\`\`

Dream:

\`\`\`bash
python scripts/run_generation.py \
  --adapter dream \
  --model Dream-org/Dream-v0-Instruct-7B \
  --input data/processed/gsm8k.jsonl \
  --output runs/pilot_dream_gsm8k.jsonl \
  --limit 50
\`\`\`

## Scientific guardrails

1. **No invented results.**
2. **Freeze before scale:** validate 50 examples on one model before the full matrix.
3. **No test-label leakage:** fit calibrators/thresholds only on a calibration partition.
4. **Store raw trajectory summaries:** do not silently redefine features after seeing test outcomes.
5. **Matched compute for RQ3:** compare adaptive policies to fixed schedules at equal or explicitly reported mean compute.
6. **Separate correctness from confidence:** labels never enter the pre-outcome feature record.
7. **Pin upstream code/model/data revisions.**

## Upstream code

TrajUQ does **not** vendor FLARE, SDAR, LLaDA, or Dream source. See \`docs/UPSTREAMS.md\` and \`THIRD_PARTY.md\`. This is especially important for FLARE/HybridDiffusion, whose authored code/checkpoints are released under PolyForm Noncommercial 1.0.0.

## Paper

Working title:

> **When Diffusion Language Models Are Unsure: Trajectory-Aware Uncertainty for Selective and Adaptive Generation**

The immediate gate is one model × 50 examples with deterministic trajectory JSONL, manual audit, stable feature hashes, no NaN/Inf, and sufficient risk variation. Only then scale.
