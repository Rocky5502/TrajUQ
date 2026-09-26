# TrajUQ

**Trajectory-Aware Uncertainty for Diffusion Language Models**

TrajUQ studies whether the *denoising trajectory itself* can predict when a diffusion language model is likely to be wrong, and whether that uncertainty can support calibrated selective generation and adaptive inference-time compute.

> **Status (2026-09-26): pre-experiment research implementation.**  
> The repository contains schema-v4 trajectory features, calibration and selective-generation metrics, leak-safe data/scoring, Dream/LLaDA/SDAR pilot adapters with parity gates, FLARE diagnostic bridging, run audits, paired bootstrap utilities, and a two-budget adaptive-compute evaluator. **No paper result is claimed yet.**

## Research questions

- **RQ1 — Predictive signal:** Which denoising-trajectory signals predict incorrect answers beyond per-token decision-time confidence?
- **RQ2 — Selective generation:** Does trajectory-aware uncertainty improve calibration and risk–coverage behavior across model families/tasks?
- **RQ3 — Adaptive compute:** Can calibrated risk allocate extra denoising/remasking steps more effectively than fixed schedules or raw-confidence heuristics?
- **RQ4 — Failure source:** How much risk is associated with denoiser ambiguity versus sampler instability, and when available, does AR/diffusion disagreement reveal failures missed by a single path?

## Core signal

For each denoising trajectory, TrajUQ records compact pre-outcome statistics rather than full vocabulary tensors. Schema v4 aggregates predictive statistics only over positions that are active and unresolved at that step; prompt tokens, future inactive blocks, and already committed tokens are excluded:

- normalized predictive entropy;
- top-1 confidence and top-2 margin;
- exact consecutive-step Jensen-Shannon drift on eligible positions;
- top-1 proposal revisions and late revisions, counted only while a position remains unresolved;
- mask persistence / commitment time;
- early-to-late entropy and revision trends;
- per-token decision-time entropy/confidence/margin at the last unresolved observation before commitment.

The transparent training-free score in `src/trajuq/risk.py` is a **ranking statistic**, not an error probability. Probability estimates must be fitted on a held-out calibration partition with `RiskCalibrator`.

## Repository layout

```text
TrajUQ/
├── configs/
│   ├── experiments/{pilot,primary}.yaml
│   └── models/{dream,llada,sdar,flare}.yaml
├── data/
│   ├── manifests/{benchmarks,evaluation_splits}.json
│   ├── samples/synthetic_smoke.jsonl
│   └── README.md
├── docs/
│   ├── ANALYSIS_PLAN.md
│   ├── DATA_SPLITS.md
│   ├── GPU_PILOT.md
│   ├── PROTOCOL.md
│   ├── REPRODUCIBILITY.md
│   ├── SDAR_INSTRUMENTATION.md
│   └── UPSTREAMS.md
├── scripts/
│   ├── prepare_eval_pair.py
│   ├── run_generation.py
│   ├── audit_run.py
│   ├── score_outputs.py
│   ├── evaluate_run.py
│   ├── compare_methods.py
│   ├── evaluate_adaptive.py
│   ├── freeze_manifest.py
│   ├── preflight.py
│   └── environment_snapshot.py
├── src/trajuq/
│   ├── adapters/
│   ├── adaptive.py
│   ├── calibration.py
│   ├── datasets.py
│   ├── decision.py
│   ├── evaluation.py
│   ├── risk.py
│   ├── schema.py
│   ├── statistics.py
│   └── trajectory.py
├── tests/
├── EXPERIMENT_FREEZE_CHECKLIST.md
├── THIRD_PARTY.md
├── CITATION.cff
├── LICENSE
└── pyproject.toml
```

## Model matrix

| Family | Public checkpoint | Role | Integration state |
|---|---|---|---|
| LLaDA-MoE | `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` | primary masked-diffusion family | sampler-aligned pilot adapter; exact GPU parity required |
| SDAR | `JetLM/SDAR-1.7B-Chat` | primary block-diffusion family | instrumented pilot adapter; on-GPU upstream parity required |
| Dream | `Dream-org/Dream-v0-Instruct-7B` | primary/extension family | hook-based pilot adapter; GPU audit required |
| FLARE / HybridDiffusion | `yuchen-zhu-zyc/HybridDiffusion-2B` | same-checkpoint AR/diffusion diagnostic | native trace bridge implemented; parity + AR/diff export pending |

Exact model/tokenizer revisions must be pinned before primary experiments. A model name alone is not a reproducible experimental specification.

## Data and split policy

Primary suite: GSM8K, MATH-500, GPQA-Diamond, ARC-Challenge, MMLU-Pro, and TruthfulQA.

Raw benchmark snapshots are **not committed**. GPQA is gated, dataset licenses/terms differ, and local caches are not reliable provenance records. TrajUQ commits identifiers, split policy, and deterministic preparation code.

Preferred split policy:

| Dataset | Calibration | Final evaluation |
|---|---|---|
| GSM8K | official train | official test |
| ARC-Challenge | official validation | official test |
| MMLU-Pro | official validation | official test |
| GPQA-Diamond | frozen internal holdout | frozen internal holdout |
| TruthfulQA | frozen internal holdout of validation | frozen internal holdout of validation |
| MATH-500 | frozen internal holdout only | frozen internal holdout only |

For one-split datasets, results are explicitly described as **internal held-out evaluation**, not a standard full-benchmark score. MATH-500 uses a pinned judge-free Math-Verify symbolic-equivalence scorer; it remains an internal held-out evaluation rather than the standard full-benchmark score.

See `docs/DATA_SPLITS.md`.

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

## Prepare a frozen benchmark pair

Install model/data extras, then pin an exact dataset revision:

```bash
pip install -e ".[models]"

python scripts/prepare_eval_pair.py \
  --dataset gsm8k \
  --revision <EXACT_DATASET_REVISION>
```

This writes:

```text
data/processed/gsm8k/
├── calibration.jsonl
├── test.jsonl
└── split_manifest.json
```

The processed benchmark directory is gitignored. The split manifest records requested revision, dataset fingerprints, counts, and SHA-256 hashes.

## Preflight and freeze

Set an exact `revision` in the selected model YAML, then:

```bash
python scripts/preflight.py \
  --model-config configs/models/llada.yaml \
  --data data/processed/gsm8k/calibration.jsonl \
  --require-cuda

python scripts/environment_snapshot.py \
  --output results/environment.json

python scripts/freeze_manifest.py \
  --model-config configs/models/llada.yaml \
  --dataset data/processed/gsm8k/calibration.jsonl \
  --dataset data/processed/gsm8k/test.jsonl \
  --output results/llada_gsm8k_freeze.json
```

A primary experiment should not use an unpinned model revision.

## First pilot choice

The first GPU pilot is frozen to **LLaDA-MoE-7B-A1B-Instruct on GSM8K calibration data**. Resolve immutable Hugging Face SHAs with `scripts/resolve_hf_revision.py`, pass the 20-prompt exact parity gate, and only then run the 50-example pilot. Full commands are in `docs/PILOT_RUNBOOK.md`.

## One-command Windows pilot gate

For the Windows GPU workstation, run the safety wrapper in stages:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage prepare
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage parity
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage pilot
```

`-Stage all` runs the same stages sequentially, but the script hard-stops if exact token parity fails. The pilot stage also writes `results/llada_gsm8k_pilot50_gate.json` and a stratified 10-example manual trace-review sheet.

## 50-example GPU pilot

```bash
python scripts/run_generation.py \
  --adapter llada \
  --model inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --revision <EXACT_MODEL_COMMIT> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/llada_gsm8k_cal.raw.jsonl \
  --limit 50 \
  --steps 128 \
  --block-length 128
```

Audit the **raw** artifact before it can see outcomes:

```bash
python scripts/audit_run.py \
  --run runs/llada_gsm8k_cal.raw.jsonl \
  --stage raw \
  --require-revision
```

Then score into a separate immutable artifact:

```bash
python scripts/score_outputs.py \
  --run runs/llada_gsm8k_cal.raw.jsonl \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/llada_gsm8k_cal.scored.jsonl

python scripts/audit_run.py \
  --run runs/llada_gsm8k_cal.scored.jsonl \
  --stage scored \
  --require-revision
```

`score_outputs.py` verifies the prompt hash before joining the external correctness label, so a modified scoring input cannot silently relabel a generation.

## Held-out evaluation

After calibration and test runs are frozen and scored:

```bash
python scripts/evaluate_run.py \
  --calibration runs/llada_gsm8k_cal.scored.jsonl \
  --test runs/llada_gsm8k_test.scored.jsonl \
  --output results/llada_gsm8k_metrics.json

python scripts/compare_calibrated.py \
  --calibration runs/llada_gsm8k_cal.scored.jsonl \
  --test runs/llada_gsm8k_test.scored.jsonl \
  --n-boot 2000 \
  --output results/llada_gsm8k_nested_bootstrap.json

python scripts/evaluate_feature_ablation.py \
  --calibration runs/llada_gsm8k_cal.scored.jsonl \
  --test runs/llada_gsm8k_test.scored.jsonl \
  --output results/llada_gsm8k_feature_ablation.json

# Generate the frozen five-sample baseline with nonzero sampling temperature:
python scripts/run_repeated_samples.py \
  --adapter llada \
  --model inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --revision <EXACT_MODEL_SHA> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output-dir runs/repeated/llada_gsm8k_cal \
  --temperature 0.2

# Evaluate repeated-sampling uncertainty (repeat --cal-sample/--test-sample):
python scripts/evaluate_repeated_sampling.py \
  --cal-target runs/llada_gsm8k_cal.scored.jsonl \
  --cal-sample runs/cal_seed1.raw.jsonl \
  --cal-sample runs/cal_seed2.raw.jsonl \
  --test-target runs/llada_gsm8k_test.scored.jsonl \
  --test-sample runs/test_seed1.raw.jsonl \
  --test-sample runs/test_seed2.raw.jsonl \
  --output results/llada_gsm8k_repeated_sampling.json

# Training-free raw-risk sensitivity:
python scripts/compare_methods.py \
  --input runs/llada_gsm8k_test.scored.jsonl \
  --n-boot 2000
```

The primary nested comparison uses a strong per-token decision-time entropy baseline and paired class-stratified bootstrap intervals on identical examples. A separately reported repeated-sampling baseline is intentionally more expensive and must include its generation cost. Positive deltas always mean the trajectory-augmented model is better; AURC is lower-is-better and is sign-normalized accordingly.

## RQ3: adaptive compute

Generate frozen base-budget and refined-budget artifacts first. TrajUQ now selects the refinement threshold from out-of-fold calibrated probabilities on the calibration partition, then fits the final calibrator on all calibration examples and applies the threshold unchanged to test:

```bash
python scripts/evaluate_adaptive.py \
  --cal-base runs/cal_base.scored.jsonl \
  --cal-refined runs/cal_refined.scored.jsonl \
  --test-base runs/test_base.scored.jsonl \
  --test-refined runs/test_refined.scored.jsonl \
  --max-refine-fraction 0.50 \
  --base-compute 64 \
  --refined-compute 128 \
  --n-boot 2000
```

## SDAR pilot

The repository includes an instrumented SDAR adapter based on the pinned MIT upstream block-diffusion generator. It is **not yet cleared for paper numbers**: first run the parity protocol in `docs/SDAR_INSTRUMENTATION.md`.

Example after pinning an exact model revision:

```bash
python scripts/run_generation.py \
  --adapter sdar \
  --model JetLM/SDAR-1.7B-Chat \
  --revision <EXACT_MODEL_COMMIT> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/sdar_gsm8k_pilot.raw.jsonl \
  --limit 50 \
  --max-new-tokens 128 \
  --block-length 4 \
  --sdar-denoising-steps 4 \
  --sdar-temperature 0
```

## LLaDA native parity

The LLaDA pilot adapter now mirrors the public sampler's attention-mask, transfer-count, and Gumbel-max semantics. Run `scripts/run_llada_reference.py` and the exact comparator in `docs/LLADA_INSTRUMENTATION.md` before any LLaDA-MoE paper number.

## Dream native parity

A native no-hook reference runner is provided for the Dream parity gate. See `docs/DREAM_INSTRUMENTATION.md` and `scripts/run_dream_reference.py`; compare exact token IDs with `scripts/compare_run_parity.py` before reporting Dream results.

## HybridDiffusion native trace

The pinned upstream runtime already exposes a JSONL trace switch. TrajUQ parses it without modifying/vendoring the SGLang server. See `docs/FLARE_INSTRUMENTATION.md` and `scripts/convert_hybrid_trace.py`.

The trace contains top-k logits only, so `topk_entropy_proxy` is **not** full-vocabulary entropy and stays an RQ4 diagnostic unless a parity-safe full-logit summary is added. Frozen same-checkpoint AR/diffusion outputs can be joined with `scripts/join_hybrid_paths.py`; the resulting disagreement features contain no correctness labels.

## MATH-500 scorer

MATH-500 scoring is now wired to pinned `math-verify==0.9.0` with ANTLR 4.13.2 and precision 6. See `docs/MATH500_SCORING.md`. The scorer is judge-free and symbolic; primary MATH-500 numbers still require the manual scorer-audit gate.

## Scientific guardrails

1. **No invented results.**
2. **Outcome isolation:** raw generation contains neither references nor correctness.
3. **Freeze before scale:** one model × 50 examples before the full matrix.
4. **Freeze data roles before generation:** use official calibration/test separation when available.
5. **No test-label tuning:** features, calibrators, thresholds, and adaptive policies are selected on calibration data only.
6. **Matched compute for RQ3:** adaptive policies are compared at the same expected refinement fraction / mean compute.
7. **Paired statistics:** method comparisons use identical examples and paired stratified bootstrap intervals.
8. **Pin revisions:** model, tokenizer, upstream runtime, data, prompts, and code commit are part of the experimental specification.
9. **Do not redistribute gated data, third-party model weights, or caches.**

## Experiment gates

The repository tracks the remaining gates as GitHub issues:

- GPU pilot / 50-item gate;
- SDAR telemetry parity (adapter implemented; GPU parity still open);
- FLARE/HybridDiffusion native-trace parity and AR/diffusion export;
- Dream hook parity;
- MATH-500 scorer manual-audit gate;
- CI runner allocation.

Do not scale to the primary model × dataset matrix until the pilot gate and model-specific telemetry gates pass.

## Paper

Working title:

> **When Diffusion Language Models Are Unsure: Trajectory-Aware Uncertainty for Selective and Adaptive Generation**

The current manuscript is intentionally pre-experiment: numerical result cells remain TBD until traceable artifacts exist.
