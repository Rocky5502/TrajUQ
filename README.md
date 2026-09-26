# TrajUQ

**Trajectory-Aware Uncertainty for Diffusion Language Models**

TrajUQ studies whether the *denoising trajectory itself* can predict when a diffusion language model is likely to be wrong, and whether that uncertainty can support calibrated selective generation and adaptive inference-time compute.

> **Status (2026-09-26): pre-experiment research implementation.**  
> The repository contains the trajectory schema/features, calibration and selective-generation metrics, leak-safe data/scoring pipeline, Dream and LLaDA pilot adapters, upstream boundaries for SDAR/FLARE, run audits, paired bootstrap utilities, and a two-budget adaptive-compute evaluator. **No paper result is claimed yet.**

## Research questions

- **RQ1 — Predictive signal:** Which denoising-trajectory signals predict incorrect answers beyond final-step confidence?
- **RQ2 — Selective generation:** Does trajectory-aware uncertainty improve calibration and risk–coverage behavior across model families/tasks?
- **RQ3 — Adaptive compute:** Can calibrated risk allocate extra denoising/remasking steps more effectively than fixed schedules or raw-confidence heuristics?
- **RQ4 — Failure source:** How much risk is associated with denoiser ambiguity versus sampler instability, and when available, does AR/diffusion disagreement reveal failures missed by a single path?

## Core signal

TrajUQ records compact pre-outcome trajectory statistics rather than full logits:

- normalized predictive entropy;
- top-1 confidence and top-2 margin;
- top-1 revisions and late revisions;
- mask persistence / commitment time;
- early-to-late entropy and revision trends.

The transparent training-free score in `src/trajuq/risk.py` is a ranking statistic, not an error probability. Probability estimates must be fitted on a held-out calibration split with `RiskCalibrator`.

## Repository layout

```text
TrajUQ/
├── configs/
│   ├── experiments/{pilot,primary}.yaml
│   └── models/{dream,llada,sdar,flare}.yaml
├── data/
│   ├── manifests/benchmarks.json
│   ├── samples/synthetic_smoke.jsonl
│   └── README.md
├── docs/
│   ├── ANALYSIS_PLAN.md
│   ├── PROTOCOL.md
│   ├── REPRODUCIBILITY.md
│   └── UPSTREAMS.md
├── scripts/
│   ├── download_data.py
│   ├── prepare_benchmarks.py
│   ├── split_inputs.py
│   ├── run_generation.py
│   ├── audit_run.py
│   ├── score_outputs.py
│   ├── evaluate_run.py
│   ├── compare_methods.py
│   ├── evaluate_adaptive.py
│   ├── freeze_manifest.py
│   └── environment_snapshot.py
├── src/trajuq/
│   ├── adapters/
│   ├── adaptive.py
│   ├── calibration.py
│   ├── decision.py
│   ├── evaluation.py
│   ├── risk.py
│   ├── schema.py
│   ├── statistics.py
│   └── trajectory.py
├── tests/
├── THIRD_PARTY.md
├── EXPERIMENT_FREEZE_CHECKLIST.md
├── CITATION.cff
├── LICENSE
└── pyproject.toml
```

## Model matrix

| Family | Public checkpoint | Role | Current integration state |
|---|---|---|---|
| LLaDA-MoE | `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` | primary masked-diffusion family | pilot adapter; GPU parity audit required |
| SDAR | `JetLM/SDAR-1.7B-Chat` | primary block-diffusion family | pinned upstream boundary; telemetry integration next |
| Dream | `Dream-org/Dream-v0-Instruct-7B` | primary/extension family | hook-based pilot adapter; GPU audit required |
| FLARE / HybridDiffusion | `yuchen-zhu-zyc/HybridDiffusion-2B` | same-checkpoint AR/diffusion diagnostic | upstream boundary; noncommercial upstream terms |

Exact model revisions must be pinned before primary experiments. A model name alone is not a reproducible experimental specification.

## Data policy

Primary manifest: GSM8K, MATH-500, GPQA-Diamond, ARC-Challenge, MMLU-Pro, and TruthfulQA.

Raw benchmark snapshots are **not committed**. GPQA is gated, and dataset licenses/terms differ. The repository commits dataset identifiers/configs/splits plus deterministic preparation code. Calibration/test partitions must be frozen **before model calls**.

MATH-500 symbolic correctness is intentionally blocked until an audited symbolic-equivalence scorer is frozen; TrajUQ will not silently approximate symbolic correctness with string matching.

## Quick start

```bash
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
```

For GPU/data work:

```bash
pip install -e ".[models]"

python scripts/download_data.py --dataset gsm8k arc_challenge mmlu_pro truthfulqa
python scripts/prepare_benchmarks.py --dataset gsm8k
python scripts/split_inputs.py   --input data/processed/gsm8k.jsonl   --output-dir data/processed/gsm8k_frozen   --calibration-fraction 0.30
```

Freeze the environment/config/data before the first real run:

```bash
python scripts/environment_snapshot.py --output results/environment.json
python scripts/freeze_manifest.py   --model-config configs/models/llada.yaml   --dataset data/processed/gsm8k_frozen/calibration.jsonl   --dataset data/processed/gsm8k_frozen/test.jsonl   --output results/llada_gsm8k_freeze.json
```

Pilot generation:

```bash
python scripts/run_generation.py   --adapter llada   --model inclusionAI/LLaDA-MoE-7B-A1B-Instruct   --revision <EXACT_MODEL_COMMIT>   --input data/processed/gsm8k_frozen/calibration.jsonl   --output runs/llada_gsm8k_cal.raw.jsonl   --limit 50   --steps 128   --block-length 128
```

The raw generation file must contain **no outcomes**:

```bash
python scripts/audit_run.py \
  --run runs/llada_gsm8k_cal.raw.jsonl \
  --stage raw \
  --require-revision

python scripts/score_outputs.py   --run runs/llada_gsm8k_cal.raw.jsonl   --input data/processed/gsm8k_frozen/calibration.jsonl   --output runs/llada_gsm8k_cal.scored.jsonl

python scripts/audit_run.py --run runs/llada_gsm8k_cal.scored.jsonl --stage scored
```

After calibration/test runs are both scored:

```bash
python scripts/evaluate_run.py   --calibration runs/llada_gsm8k_cal.scored.jsonl   --test runs/llada_gsm8k_test.scored.jsonl   --output results/llada_gsm8k_metrics.json

python scripts/compare_methods.py   --input runs/llada_gsm8k_test.scored.jsonl   --n-boot 2000
```

RQ3 uses frozen base/refined compute runs. Select the threshold on calibration only, then apply it unchanged to test:

```bash
python scripts/evaluate_adaptive.py   --cal-base runs/cal_base.scored.jsonl   --cal-refined runs/cal_refined.scored.jsonl   --test-base runs/test_base.scored.jsonl   --test-refined runs/test_refined.scored.jsonl   --max-refine-fraction 0.50   --base-compute 64   --refined-compute 128
```

## Scientific guardrails

1. **No invented results.**
2. **Outcome isolation:** raw generation artifacts contain no references/correctness.
3. **Freeze before scale:** one model × 50 examples before the full matrix.
4. **No test-label tuning:** features, calibrators, thresholds, and adaptive policies are selected on calibration data only.
6. **Matched compute for RQ3:** adaptive policies are compared at the same expected refinement fraction/mean compute.
6. **Paired statistics:** method comparisons use identical examples and paired stratified bootstrap intervals.
7. **Pin revisions:** model, tokenizer, upstream runtime, data, prompts, and code commit are part of the experimental specification.
8. **Do not redistribute gated data or model weights.**

## Paper

Working title:

> **When Diffusion Language Models Are Unsure: Trajectory-Aware Uncertainty for Selective and Adaptive Generation**

The immediate experimental gate is a 50-example GPU pilot with a clean raw audit, manual trace inspection, deterministic feature recomputation, stable model/data revisions, and enough risk variation to justify scaling.
