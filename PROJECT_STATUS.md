# TrajUQ project status and continuation handoff

**Read this file first when resuming TrajUQ in a new chat or on a new machine.**
It is versioned in the repository so the project does not depend on one
conversation transcript.

## Scientific freeze

Target: **NAACL 2027 via ARR**.

Working paper title:

> **When Diffusion Language Models Are Unsure: Trajectory-Aware Uncertainty for Selective and Adaptive Generation**

The primary experiment is frozen to **3 open-source diffusion-LM families x 6
datasets = 18 model-dataset cells**:

- LLaDA-MoE: inclusionAI/LLaDA-MoE-7B-A1B-Instruct
- SDAR: JetLM/SDAR-1.7B-Chat
- Dream: Dream-org/Dream-v0-Instruct-7B
- GSM8K, MATH-500, GPQA-Diamond, ARC-Challenge, MMLU-Pro, TruthfulQA

FLARE / HybridDiffusion is a **diagnostic RQ4 extension**, not a fourth primary
family. The primary positive-claim gate is hierarchical macro AUROC improvement
over calibrated decision-time entropy, with the confidence-interval and
model-family consistency rules frozen in configs/experiments/primary.yaml.

The five-sample uncertainty baseline now covers all 18 cells. MATH-500 samples
are clustered without references using pinned symbolic equivalence. RQ3 uses a
50% primary second-pass selection cap with frozen 25% and 75% sensitivities.

## What is implemented

The repository-side scientific/engineering pipeline is largely built. In
particular, the following are implemented and versioned:

- schema-v4 active/unresolved trajectory instrumentation and frozen features;
- raw/scored outcome isolation, prompt-hash checking, and artifact audits;
- held-out risk calibration, discrimination, calibration, and tie-aware
  risk-coverage metrics;
- exact-parity harnesses for LLaDA, SDAR, and Dream, including same prompt hash,
  model/revision, seed, and adapter-specific frozen-generation provenance
  checks; primary readiness accepts only protocol-v2 parity artifacts with
  strict token and provenance checks enabled;
- frozen 18-cell primary planning and hierarchical aggregation;
- pre-registered RQ1 positive-claim gate;
- five-seed repeated-sampling baseline and full-matrix macro aggregation;
- pinned MATH-500 symbolic scoring, automatic audit, deterministic calibration
  review-sheet export, and human attestation gate;
- RQ3 adaptive-compute planning/evaluation/18-cell hierarchical aggregation;
- same-checkpoint FLARE AR/diffusion RQ4 diagnostic pipeline;
- immutable model/dataset revision ledger;
- claim-gated manuscript result rendering;
- anonymous ARR supplement builder;
- PowerShell and Bash CPU-verification entry points.

Machine-readable implementation evidence and expected runtime artifacts live in
configs/project_status.yaml. Run:

~~~bash
python scripts/project_status.py
~~~

to write results/project_status_snapshot.json and
results/project_status_snapshot.md. The dashboard reports implementation
evidence, required-vs-optional execution progress by phase, and the **next
required action** with its expected artifact. These local snapshots separate
implemented code from executed experimental gates, so a high engineering
completion count cannot be mistaken for completed empirical work.

## What is still blocked / not done

The remaining work is primarily **execution**, not paper-number invention.

### Pre-primary hard gates

1. Run the clean-commit CPU verification suite and archive
   results/cpu_verification.json. Hosted GitHub Actions is still preferred when
   runner allocation works, but the same suite is runnable locally. See issue #6.
2. Resolve immutable model/data SHAs and archive results/revision_ledger.json.
3. Run LLaDA 20-prompt native-vs-instrumented exact parity. See issue #7.
4. Run the 50-example LLaDA + GSM8K calibration pilot and complete the 10-trace
   human review. See issue #1.
5. Run SDAR exact parity. See issue #2.
6. Run Dream exact parity. See issue #5.
7. Run the automatic MATH-500 scorer audit, then a >=25-example
   calibration-only MATH-500 scoring pilot, export the deterministic review
   sheet, inspect it, and record the human scorer attestation. See issue #4.
8. Require python scripts/primary_readiness.py to return status=PASS.

Do **not** launch the full 18-cell primary matrix before these gates pass.

### Primary / analysis execution

After readiness passes:

1. prepare all six immutable dataset pairs;
2. create and archive results/primary_matrix_plan.json;
3. dry-run scripts/execute_primary_plan.py, then execute cells from the frozen
   plan; the executor writes a resumable SHA-verified execution state;
4. execute all 18 calibration/test cells exactly as planned;
5. audit and score every cell without overwriting completed artifacts;
6. create results/primary_analysis_plan.json;
7. dry-run scripts/execute_primary_analysis.py and then use it to resume all
   cell-level metrics, nested comparisons, ablations, and raw-risk diagnostics;
8. aggregate the complete 18-cell primary endpoint;
9. run the RQ1 claim gate **without changing it after seeing the result**;
10. run the complete 18-cell five-sample comparator with
   scripts/execute_repeated_sampling.py; verified seed artifacts are resumable;
11. run RQ3 base-budget generations and all frozen 25/50/75% cap analyses
   with scripts/execute_adaptive_plan.py;
12. optionally run FLARE same-checkpoint RQ4 after its parity/trace gates;
13. render paper numbers only through scripts/render_results_tex.py;
14. run scripts/submission_readiness.py to re-check required project gates,
    source hashes, anonymous paper audit, and the deterministic software
    supplement before the external ACL pubcheck/upload step.

A negative RQ1 claim gate is a valid scientific result. Do not redesign the
test analysis because a frozen result is weak.

## Current issue map

- #1 - 50-item LLaDA + GSM8K pilot gate
- #2 - SDAR native parity
- #3 - FLARE telemetry + same-checkpoint RQ4 diagnostic execution
- #4 - MATH-500 scorer execution/manual audit
- #5 - Dream native parity
- #6 - GitHub Actions runner allocation
- #7 - LLaDA native parity

Keep these issues open until their **execution artifacts** pass. Implemented
support code alone is not grounds for closing an execution gate.

## Paper state

paper/main.tex is the anonymous ARR/ACL main paper and paper/appendix.tex
carries the large appendix. The paper is intentionally pre-results.
paper/generated_results.tex contains fallback TBD macros.

Never type empirical values directly into paper/main.tex. Once the complete
frozen result artifacts exist:

~~~bash
python scripts/render_results_tex.py
python scripts/audit_paper.py --submission-ready
python scripts/submission_readiness.py
~~~

The renderer checks complete cell sets and the RQ1 claim-gate decision before
populating paper macros.

## Commands to run first on a fresh continuation

~~~bash
git pull
python scripts/project_status.py
~~~

Then run the cross-platform local verification:

~~~powershell
PowerShell -ExecutionPolicy Bypass -File scripts/verify_cpu.ps1
~~~

or:

~~~bash
bash scripts/verify_cpu.sh
~~~

Then follow docs/PILOT_RUNBOOK.md for the first GPU gate and
docs/PRIMARY_MATRIX_RUNBOOK.md after readiness passes.

## New-chat continuation prompt

A future chat can start with:

> Continue the TrajUQ NAACL 2027 project from the repository
> Rocky5502/TrajUQ. Read PROJECT_STATUS.md, configs/project_status.yaml, the
> open GitHub issues, and the latest main branch before changing anything.
> Preserve the frozen 3x6 primary design, outcome isolation, RQ1 claim gate,
> five-sample 18-cell comparator, and RQ3 50% primary cap with 25/75%
> sensitivities. Do not invent results. Continue from the first unmet execution
> gate and update PROJECT_STATUS.md whenever a milestone materially changes.

## Maintenance rule

Whenever a milestone materially changes, update all three together:

1. the relevant GitHub issue;
2. configs/project_status.yaml if the expected artifact/protocol changes;
3. this PROJECT_STATUS.md handoff file.

This makes the repository - not the chat transcript - the canonical project
memory.
