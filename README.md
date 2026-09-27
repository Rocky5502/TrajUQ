# TrajUQ

**Trajectory-Aware Uncertainty for Diffusion Language Models**

> **Continuation / handoff:** read `PROJECT_STATUS.md` first. It tracks the
> frozen design, what is implemented, the exact execution gates still pending,
> and the safest next action if work resumes in another chat.

TrajUQ studies whether the *denoising trajectory itself* can predict when a diffusion language model is likely to be wrong, and whether that uncertainty can support calibrated selective generation and adaptive inference-time compute.

> **Status (2026-09-27): pre-experiment research implementation.**  
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
│   ├── A100_EXECUTION.md
│   ├── ANALYSIS_PLAN.md
│   ├── DATA_SPLITS.md
│   ├── GPU_PILOT.md
│   ├── PROTOCOL.md
│   ├── REPRODUCIBILITY.md
│   ├── RESULT_ARTIFACT_MAP.md
│   ├── SDAR_INSTRUMENTATION.md
│   └── UPSTREAMS.md
├── scripts/
│   ├── prepare_eval_pair.py
│   ├── run_generation.py
│   ├── audit_run.py
│   ├── score_outputs.py
│   ├── evaluate_run.py
│   ├── compare_methods.py
│   ├── aggregate_feature_ablation.py
│   ├── a100_campaign_linux.sh
│   ├── build_overleaf_package.py
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
| Dream | `Dream-org/Dream-v0-Instruct-7B` | primary/extension family | pre-transfer logits-hook telemetry; exact GPU parity required |
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

## Project status dashboard

Run:

```bash
python scripts/project_status.py
```

This verifies declared implementation evidence and inspects local experiment
artifacts without pretending that implemented code means completed experiments.
The canonical handoff is `PROJECT_STATUS.md`; the machine-readable gate
specification is `configs/project_status.yaml`. The result-to-paper bridge is
`docs/RESULT_ARTIFACT_MAP.md`, which fixes where every required frozen result
enters the manuscript and prevents manual result copying.

## Local verification and frozen pre-primary commit

The hosted Actions job is currently failing before runner allocation. On Windows,
the preferred pre-primary command is:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/freeze_preprimary_windows.ps1
```

It requires a clean worktree, runs the complete CPU verification suite, resolves
all primary model/data repositories to immutable Hugging Face SHAs, verifies
record-level access to every gated primary dataset at those exact revisions,
and writes `results/revision_ledger.json`,
`results/gated_dataset_access.json`, `results/dataset_contracts.json`, plus
hash-bound configs in `results/frozen_models/`. The CPU artifact, revision
ledger, gated-access proof, and six-dataset contract proof must record the same
clean git commit. Both data preflights bind the revision-ledger SHA-256.
`dataset_contracts.json` streams four exact-revision rows from each primary
dataset, exercises the frozen normalizer/scorer contract on every sampled row,
stores no benchmark row content, and is accepted only when all four rows pass.

Linux/macOS guarded equivalent:

```bash
bash scripts/freeze_preprimary.sh
```

Both freeze wrappers require a clean checkout, run CPU verification, immutable
revision resolution, exact-revision gated-data access verification, and the
live six-dataset normalizer/scorer contract smoke without changing commits,
re-check cleanliness afterward,
refresh the project-status snapshot, and export a privacy-scanned private
pre-primary handoff bundle. Primary readiness v2 later requires all three
parity-v3 artifacts to match this ledger git commit and their ledger-resolved
model SHA. CPU verification covers compile checks, Ruff, pytest, metric smoke,
the leak-safe synthetic end-to-end pipeline, the pinned MATH-500 scorer audit,
and the paper anonymity audit. It does not replace GPU model parity.

### NVIDIA A100 80GB Linux path

For the server run, use the canonical end-to-end checklist in
`docs/A100_EXECUTION.md`. It covers environment validation, the immutable
pre-primary freeze, LLaDA parity + 50-example GSM8K pilot, the human pilot
attestation, SDAR/Dream parity, the MATH-500 automatic/manual scorer gates,
primary readiness, safe preparation of only missing dataset pairs, the frozen
18-cell matrix, primary analysis, the five-seed RQ2 comparator, RQ3 adaptive
compute, and final artifact-derived paper rendering.

The key rule is simple: after `freeze_preprimary.sh` passes, do not edit tracked
experiment code on that run. A required code repair means a new commit and a new
freeze; never repair code underneath already-produced frozen evidence.

The staged convenience runner is:

```bash
bash scripts/a100_campaign_linux.sh status
bash scripts/a100_campaign_linux.sh pilot
# human 10-trace review + attestation
bash scripts/a100_campaign_linux.sh parity-math
# human 25-example MATH-500 review + attestation
bash scripts/a100_campaign_linux.sh readiness
bash scripts/a100_campaign_linux.sh estimate
cat results/a100_compute_estimate.md
```

For the predeclared three-day single-A100 rental, once readiness is PASS run:

```bash
BUDGET_HOURS=72 MONITOR_SECONDS=3600 \
  bash scripts/a100_campaign_linux.sh autopilot-72h
```

The 72-hour supervisor runs the resumable stages in scientific-priority order
(primary, analysis, RQ3, expensive five-sample RQ2, paper rendering), writes an
hourly GPU/disk/progress log under `logs/a100_72h/`, retries a failed resumable
stage once, and stops rather than weakening a scientific gate if the wall-clock
budget is exhausted. It does not require an AI agent to remain connected while
the GPU process is healthy.

The two human-review boundaries are deliberate and cannot be bypassed by either
campaign runner.

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

The processed benchmark directory is gitignored. The split manifest records the
requested revision, dataset fingerprints, uncapped source counts, frozen
pre-generation cap policy, selected counts, and SHA-256 hashes. The current
three-day profile uses outcome-free SHA-256 ID ranking with seed 2027:
GSM8K calibration/test are capped at 500 each; ARC-Challenge and MMLU-Pro
retain their smaller calibration splits and cap the larger test split at 500;
GPQA-Diamond, TruthfulQA, and MATH-500 keep their complete frozen holdouts.
The selection rule never consults references, correctness, prompt text, or
model outputs.

## Preflight and freeze

Keep the tracked model/data configs as generic templates. Before **any primary
model parity run**, freeze one clean experiment commit and run
`python scripts/resolve_experiment_revisions.py` (or the Windows freeze wrapper).
Revision-ledger protocol v2 records the experiment git commit, hashes of the
primary config and dataset manifest, immutable SHAs for all three primary models
and all six datasets, and hashes of the artifact-local frozen model configs.
The lower-level `scripts/resolve_hf_revision.py` and
`scripts/freeze_model_config.py` remain debugging tools, not the primary
freeze path. Use the ledger-generated frozen config for preflight and manifests:

```bash
python scripts/preflight.py \
  --model-config results/frozen_models/llada.yaml \
  --data data/processed/gsm8k/calibration.jsonl \
  --require-cuda

python scripts/environment_snapshot.py \
  --output results/environment.json

python scripts/freeze_manifest.py \
  --model-config results/frozen_models/llada.yaml \
  --dataset data/processed/gsm8k/calibration.jsonl \
  --dataset data/processed/gsm8k/test.jsonl \
  --output results/llada_gsm8k_freeze.json
```

A primary experiment should not use an unpinned model revision.

## First pilot choice

The first GPU pilot is frozen to **LLaDA-MoE-7B-A1B-Instruct on GSM8K
calibration data**. Freeze the clean experiment commit and immutable revisions
first, pass the 20-prompt exact parity-v3 gate at that same commit/model SHA, and
only then run the 50-example pilot. Full commands are in
`docs/PILOT_RUNBOOK.md`.

## Windows pre-primary + pilot gate

For the Windows GPU workstation, run:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/freeze_preprimary_windows.ps1
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage prepare
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage parity
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage pilot
```

The freeze wrapper writes `results/cpu_verification.json`,
`results/revision_ledger.json`, and `results/frozen_models/*.yaml`. The pilot
wrapper consumes those frozen LLaDA/GSM8K revisions; it does not re-resolve a
moving Hugging Face head. Before every GPU parity/pilot stage it revalidates
the frozen GSM8K split, split audit, model config, and freeze-manifest hashes.
The parity stage hard-stops unless protocol v3 has exactly 20 shared prompts,
zero output/provenance mismatches, and the same git commit/model SHA as the
ledger. The pilot stage requires both raw/scored `audit_run.py` artifacts to
remain PASS and SHA-bound before existing evidence can be reused. It writes
`results/llada_gsm8k_pilot50_gate.json` plus a stratified 10-example manual
trace-review sheet that records the exact scored-pilot path and SHA-256.

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

Audit the **raw** artifact before it can see outcomes. The audit now checks schema/version, frozen feature presence, semantic bounds for entropy/confidence/JS/rate features, per-step summaries, v4 trajectory semantics, and recursively rejects outcome-like metadata keys:

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
  --output results/llada_gsm8k_metrics.json \
  --predictions-output results/llada_gsm8k_predictions.jsonl

pip install -e ".[analysis]"
python scripts/plot_paper_figures.py \
  --predictions results/llada_gsm8k_predictions.jsonl \
  --output-dir paper/figures/generated

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

# Run the same five seeds for calibration and test.
python scripts/run_repeated_samples.py \
  --adapter llada \
  --model inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --revision <EXACT_MODEL_SHA> \
  --input data/processed/gsm8k/test.jsonl \
  --output-dir runs/repeated/llada_gsm8k_test \
  --temperature 0.2

# Then evaluate, supplying the five seed files from each directory.
python scripts/evaluate_repeated_sampling.py \
  --cal-target runs/llada_gsm8k_cal.scored.jsonl \
  --cal-sample runs/repeated/llada_gsm8k_cal/seed-2027.raw.jsonl \
  --cal-sample runs/repeated/llada_gsm8k_cal/seed-2028.raw.jsonl \
  --cal-sample runs/repeated/llada_gsm8k_cal/seed-2029.raw.jsonl \
  --cal-sample runs/repeated/llada_gsm8k_cal/seed-2030.raw.jsonl \
  --cal-sample runs/repeated/llada_gsm8k_cal/seed-2031.raw.jsonl \
  --test-target runs/llada_gsm8k_test.scored.jsonl \
  --test-sample runs/repeated/llada_gsm8k_test/seed-2027.raw.jsonl \
  --test-sample runs/repeated/llada_gsm8k_test/seed-2028.raw.jsonl \
  --test-sample runs/repeated/llada_gsm8k_test/seed-2029.raw.jsonl \
  --test-sample runs/repeated/llada_gsm8k_test/seed-2030.raw.jsonl \
  --test-sample runs/repeated/llada_gsm8k_test/seed-2031.raw.jsonl \
  --expected-samples 5 \
  --output results/llada_gsm8k_repeated_sampling.json \
  --predictions-output results/llada_gsm8k_repeated_predictions.jsonl
```

The frozen repeated-sampling baseline requires exactly five seeds
(2027--2031) on the pre-registered 12-cell subset: all three primary models crossed with GSM8K, MATH-500, GPQA-Diamond, and MMLU-Pro. The evaluator verifies unique/matched seeds, identical
generation settings between calibration and test, raw outcome isolation, model
identity, prompt hashes, and reports the auxiliary repeated-generation wall
time. The frozen 12-cell representative aggregator emits protocol
`trajuq_vs_repeated_macro_v2`; it refuses incomplete test-time wall-clock
metadata and reports the equal-cell-weight mean auxiliary seconds per example
with target generation excluded. After both prediction artifacts exist, compare
the expensive baseline directly with TrajUQ using:

```bash
python scripts/compare_repeated_to_trajuq.py \
  --trajuq-predictions results/llada_gsm8k_predictions.jsonl \
  --repeated-predictions results/llada_gsm8k_repeated_predictions.jsonl \
  --output results/llada_gsm8k_trajuq_vs_repeated.json
```

Training-free raw-risk sensitivity:

```bash
python scripts/compare_methods.py \
  --input runs/llada_gsm8k_test.scored.jsonl \
  --n-boot 2000
```

The primary nested comparison uses a strong per-token decision-time entropy baseline and paired class-stratified bootstrap intervals on identical examples. After all model-dataset cells are complete, use `scripts/aggregate_cells.py` to macro-average cells without pooling heterogeneous examples. A separately reported repeated-sampling baseline is intentionally more expensive and must include its generation cost. Positive deltas always mean the trajectory-augmented model is better; AURC is lower-is-better and is sign-normalized accordingly.

## RQ3: adaptive compute

Generate frozen base-budget and refined-budget artifacts first. TrajUQ now selects the refinement threshold from out-of-fold calibrated probabilities on the calibration partition, then fits the final calibrator on all calibration examples and applies the threshold unchanged to test:

```bash
python scripts/evaluate_adaptive.py \
  --cal-base runs/cal_base.scored.jsonl \
  --cal-refined runs/cal_refined.scored.jsonl \
  --test-base runs/test_base.scored.jsonl \
  --test-refined runs/test_refined.scored.jsonl \
  --max-refine-fraction 0.50 \
  --base-compute 32 \
  --refined-compute 128 \
  --n-boot 2000 \
  --output results/llada_gsm8k_adaptive.json \
  --predictions-output results/llada_gsm8k_adaptive.predictions.jsonl
```

For the full experiment, do not hand-build these commands. Run
`scripts/plan_adaptive_runs.py`; it derives the base runs from the frozen
18-cell primary plan and emits evaluations for the 50% primary cap plus the
25%/75% sensitivities. Its `aggregate_adaptive.py` commands produce
equal-cell-weight hierarchical macro summaries.

## SDAR pilot

The SDAR hard parity gate now has a pinned-upstream reference runner (`scripts/run_sdar_reference.py`). Because upstream SDAR always uses multinomial sampling, parity is seed-matched at its native `temperature=1.0`; `temperature=0` remains a separate TrajUQ argmax research control.


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
  --temperature 1.0
```

## LLaDA native parity

The LLaDA pilot adapter now mirrors the public sampler's attention-mask, transfer-count, and Gumbel-max semantics. Run `scripts/run_llada_reference.py` and the exact comparator in `docs/LLADA_INSTRUMENTATION.md` before any LLaDA-MoE paper number.

## Dream native parity

A native no-hook reference runner is provided for the Dream parity gate. See `docs/DREAM_INSTRUMENTATION.md` and `scripts/run_dream_reference.py`; compare exact token IDs with `scripts/compare_run_parity.py` before reporting Dream results.

## HybridDiffusion native trace

The pinned upstream runtime already exposes a JSONL trace switch. TrajUQ parses it without modifying/vendoring the SGLang server. See `docs/FLARE_INSTRUMENTATION.md` and `scripts/convert_hybrid_trace.py`.

The trace contains top-k logits only, so `topk_entropy_proxy` is **not** full-vocabulary entropy and stays an RQ4 diagnostic unless a parity-safe full-logit summary is added. Frozen same-checkpoint AR/diffusion outputs can be joined with `scripts/join_hybrid_paths.py`; the resulting disagreement features contain no correctness labels.

## MATH-500 scorer

MATH-500 scoring is wired to pinned `math-verify==0.9.0` with ANTLR 4.13.2 and precision 6. The five-sample baseline uses the same equivalence relation for reference-free pairwise clustering, so MATH-500 remains in the frozen 12-cell representative comparator. See `docs/MATH500_SCORING.md`. Before the primary matrix, both `scripts/audit_math500_scorer.py` and a 25-example calibration-split human review must pass. Use `scripts/math500_gate_windows.ps1 -Stage all` on Windows for the guarded automated path. The deterministic review sheet records the exact scored-pilot and frozen calibration-input SHA-256 values; the human recorder and primary readiness revalidate those sources plus the scored-pilot audit before accepting the attestation.

## Scientific guardrails

1. **No invented results.**
2. **Outcome isolation:** raw generation contains neither references nor correctness.
3. **Freeze before scale:** one model × 50 examples before the full matrix.
4. **Freeze data roles before generation:** use official calibration/test separation when available.
5. **No test-label tuning:** features, calibrators, thresholds, and adaptive policies are selected on calibration data only.
6. **Matched RQ3 comparator:** adaptive selection is compared with random second-pass selection at the same selected fraction; actual second-pass wall time is also reported when available.
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

Do not scale to the primary model × dataset matrix until the clean CPU artifact,
revision-ledger v2, gated-data access v1, six-dataset contract v1,
LLaDA/SDAR/Dream parity-v3 gates, 50-item pilot + human review, and MATH-500
automatic/manual scorer gates are complete. Then run
`python scripts/primary_readiness.py`. Readiness v2 rejects stale or mixed
artifacts: CPU verification and all parity files must share the ledger git
commit; parity source files are re-hashed; pilot raw/scored audits must still
bind the pilot files; the pilot review must identify the exact scored source;
and the MATH review must still bind the frozen calibration input, scored pilot,
and scored audit. Only after readiness PASS should
`python scripts/plan_primary_matrix.py` create the inspectable 18-cell
execution plan.

## Overleaf package

Build a clean anonymous Overleaf ZIP from the current paper source and the
official ACL style pinned at commit
`d5adc823ff0f80f98c80405ca0ab66c68e684409`:

```bash
python scripts/build_overleaf_package.py
```

The builder requires a clean checkout, scans paper/style text for identity and
local-path leaks, includes generated result PDFs when they exist, and writes a
deterministic ZIP with an internal SHA-256 manifest.

## Paper

Working title:

> **When Diffusion Language Models Are Unsure: Trajectory-Aware Uncertainty for Selective and Adaptive Generation**

The current manuscript is intentionally pre-experiment: numerical result cells remain TBD until traceable artifacts exist.


## Cross-cell macro aggregation

Once each model-dataset cell has exported held-out probabilities with
`evaluate_run.py --predictions-output`, aggregate them with equal cell weight:

```bash
python scripts/aggregate_cells.py \
  --cell llada_gsm8k=results/llada_gsm8k_predictions.jsonl \
  --cell dream_gsm8k=results/dream_gsm8k_predictions.jsonl \
  --cell sdar_gsm8k=results/sdar_gsm8k_predictions.jsonl \
  --n-boot 2000 \
  --output results/macro_primary.json
```

The bootstrap resamples examples within every cell, preserves pairing between
TrajUQ and the decision-entropy baseline, then macro-averages cells. It does not
pool examples from differently sized datasets. For primary reporting the
aggregator requires the complete frozen 3×6 cell set; `--allow-partial` is
exploratory only. Apply the pre-registered positive-claim wording gate with
`scripts/evaluate_rq1_claim_gate.py` after macro aggregation.


## Anonymous ARR software supplement

Do not submit or link the public development repository during anonymized review.
Build a deterministic software ZIP instead:

```bash
python scripts/build_anonymous_supplement.py
```

The builder excludes paper source, author metadata, raw/gated data, model weights,
run outputs, and the public development-repository identity. It scans selected
text files for identity/tracking-path leaks and writes a SHA-256 manifest inside
the archive.


## Full primary matrix

The tracked YAML files remain generic templates. Freeze one clean experiment commit with CPU verification + revision-ledger v2
+ gated-access v1 + six-dataset contract v1, execute the model parity, pilot,
and MATH-500 scorer gates at that frozen provenance, and require
`scripts/primary_readiness.py` to PASS. Prepare all six dataset pairs from the
ledger SHAs and only then generate `results/primary_matrix_plan.json` with
`scripts/plan_primary_matrix.py`. See `docs/PRIMARY_MATRIX_RUNBOOK.md`. The
companion `scripts/plan_adaptive_runs.py` derives all frozen low-budget RQ3
runs from that primary plan, so base/refined paths and compute accounting cannot
drift manually.
