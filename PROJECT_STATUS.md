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
  checks; parity protocol v3 records uniform run-level provenance, clean-checkout
  state, and SHA-256 bindings to both reference and instrumented JSONL; primary
  readiness re-hashes that evidence and accepts only exact 20-prompt artifacts
  bound to the frozen revision-ledger git commit and model SHA;
- frozen 18-cell primary planning and hierarchical aggregation; the primary execution plan is hash-bound to the experiment commit, configs, frozen model configs, and prepared dataset inputs;
- pre-registered RQ1 positive-claim gate;
- five-seed repeated-sampling baseline and full-matrix macro aggregation; repeated-sampling plan protocol v2 binds the primary/analysis plans, config, scored targets, prompt partitions, and TrajUQ prediction artifacts by SHA-256 before auxiliary generation; repeated macro protocol v2 additionally requires complete test-time auxiliary wall-clock metadata in all 18 cells and carries an equal-cell-weight compute summary into artifact-derived paper rendering;
- pinned MATH-500 symbolic scoring, commit-bound automatic audit with runtime
  dependency-version verification, deterministic calibration review-sheet
  export, a guarded calibration-only Windows pilot wrapper, and SHA-bound human
  attestation gate;
- RQ3 adaptive-compute planning/evaluation/18-cell hierarchical aggregation; adaptive plan protocol v3 binds the primary plan/config plus refined scored artifacts and their audits before low-budget execution;
- same-checkpoint FLARE AR/diffusion RQ4 diagnostic pipeline;
- revision-ledger protocol v2 binding the experiment git commit, primary/data
  config hashes, immutable model/data SHAs, and frozen model-config hashes;
- claim-gated manuscript result rendering;
- anonymous ARR supplement builder;
- PowerShell and Bash CPU-verification entry points plus one-command strict
  pre-primary freeze wrappers for Windows and Linux/macOS;
- a guarded Windows parity wrapper for SDAR/Dream that consumes the frozen
  revision ledger, enforces a clean matching checkout, preserves valid PASS
  artifacts, and refuses ambiguous partial evidence;
- protocol-v1 raw/scored run audits now record the clean git checkout and commit;
  pre-primary parity, pilot, human-review, MATH-500, readiness, primary execution,
  primary analysis, repeated sampling, and adaptive compute all re-hash those
  audits against their source artifacts and require the frozen experiment
  commit, preventing a stale PASS label from being reused anywhere in the
  primary evidence chain;
- pilot and MATH-500 human review sheets are source-bound to the exact scored
  calibration artifacts they display; their recorders and primary readiness
  independently revalidate those hashes before the 18-cell matrix can launch;
- clean-checkout provenance guards across the pilot wrapper, primary readiness,
  all frozen planners, and all primary/repeated/adaptive executors, so an
  uncommitted source edit cannot masquerade under the frozen git SHA;
- primary-readiness v2 additionally revalidates the revision-ledger dataset
  manifest hash, the exact CPU-verification protocol/step sequence, raw/scored
  pilot audit SHA bindings, clean pilot-quality and MATH-500 automatic-audit
  provenance, frozen MATH dependency versions, source-bound clean-checkout
  human attestations, MATH scored-pilot audit evidence, and hash-bound parity sources;
- a versioned `NEW_CHAT_MASTER_PROMPT.md` plus a fail-closed private research
  handoff exporter that refuses to omit required continuation context;
- guarded Linux/A100 wrappers for the LLaDA prepare/parity/pilot path,
  remaining SDAR/Dream parity gates, and the MATH-500 scorer pilot, so the full
  pre-primary GPU path can run on one frozen Linux checkout without weakening
  the Windows evidence contract;
- a canonical `docs/A100_EXECUTION.md` runbook that carries the A100 server
  from environment validation through the immutable freeze, pre-primary gates,
  18-cell execution, RQ2/RQ3 follow-ons, and artifact-derived paper rendering
  without regenerating already-observed frozen data.

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

1. Select one clean experiment commit. On Windows run
   `scripts/freeze_preprimary_windows.ps1`; it runs clean CPU verification and
   revision-ledger v2 resolution without changing commits. Hosted GitHub Actions
   remains preferred when runner allocation works, but issue #6 still tracks the
   current runner-allocation failure.
2. Require `results/cpu_verification.json` PASS and
   `results/revision_ledger.json` protocol-v2 PASS at the same git commit.
3. Run LLaDA 20-prompt native-vs-instrumented parity protocol v3 at the
   ledger-resolved LLaDA SHA. Require exactly 20 shared prompts, zero output
   mismatches, zero provenance mismatches, a clean comparator checkout, and
   unchanged reference/instrumented SHA-256 evidence. The Windows wrapper now
   preserves a valid PASS and refuses partial/stale evidence instead of deleting
   artifacts. See issue #7.
4. Run the 50-example LLaDA + GSM8K calibration pilot. Pilot-quality protocol v2
   binds raw/scored files by SHA-256, records a clean checkout, and binds the
   frozen git commit/model SHA. Primary readiness also requires both raw/scored
   `audit_run.py` artifacts to PASS and still match those files. Inspect at
   least 10 exported trajectories; the review sheet now records the exact scored
   artifact path/SHA, and the protocol-v2 human attestation is accepted only
   when that source binding, parity artifact, and pilot-quality artifact remain
   unchanged on the same frozen clean checkout. See issue #1.
5. Run SDAR exact parity protocol v3 at the same frozen commit/revision. On
   Windows, use `scripts/parity_remaining_windows.ps1 -Model sdar`; it binds
   the run to the clean CPU/revision freeze and refuses ambiguous partial
   artifacts. See issue #2.
6. Run Dream exact parity protocol v3 at the same frozen commit/revision. On
   Windows, use `scripts/parity_remaining_windows.ps1 -Model dream` with the
   same frozen-state and non-overwrite safeguards. See issue #5.
7. Require the MATH-500 automatic scorer audit protocol v2 at the frozen clean
   commit, then a >=25-example calibration-only MATH-500 scoring pilot,
   deterministic review sheet, and protocol-v2 human attestation. The wrapper
   revalidates the frozen MATH split/audit and LLaDA parity source hashes before
   running. The review sheet records the exact scored-pilot and calibration-input
   SHA-256 values; the recorder and readiness independently revalidate those
   sources plus the scored-pilot audit. On Windows,
   `scripts/math500_gate_windows.ps1 -Stage all` performs the automated portion
   and deliberately stops before the human attestation. Readiness also requires
   math-verify=0.9.0 and antlr4-python3-runtime=4.13.2. See issue #4.
8. Run `python scripts/primary_readiness.py` and require protocol
   `trajuq_primary_readiness_v2` with `status=PASS`. It rejects stale/mixed
   CPU, parity, pilot raw/scored audits, pilot review sources, MATH-500
   scorer/dependency/review sources, dataset-manifest, commit, and dirty-worktree
   state.

Do **not** launch the full 18-cell primary matrix before these gates pass.

### Primary / analysis execution

After readiness passes:

1. prepare all six immutable dataset pairs;
2. create and archive results/primary_matrix_plan.json;
3. dry-run scripts/execute_primary_plan.py, then execute cells from the frozen
   plan; reusable raw/scored evidence must have clean protocol-v1 audits from
   the frozen experiment commit, and the executor writes a resumable
   SHA-verified execution state;
4. execute all 18 calibration/test cells exactly as planned;
5. audit and score every cell without overwriting completed artifacts;
6. create results/primary_analysis_plan.json only after all 18 scored artifacts
   have clean protocol-v1 scored audits from the frozen experiment commit;
   analysis-plan protocol v2 then SHA-binds those inputs, audits, the primary
   plan/config, and experiment commit;
7. dry-run scripts/execute_primary_analysis.py and then use it to resume all
   cell-level metrics, nested comparisons, ablation summaries plus per-example
   ablation probabilities, and raw-risk diagnostics; the executor refuses
   upstream provenance drift before reading result values;
8. aggregate both the complete 18-cell primary endpoint and the complete frozen
   feature-ablation macro with equal-cell paired hierarchical bootstrap;
9. run the RQ1 claim gate **without changing it after seeing the result**;
10. freeze repeated-sampling plan protocol v2 after primary analysis predictions
    exist, then run the complete 18-cell five-sample comparator with
    scripts/execute_repeated_sampling.py; primary scored targets and every reused
    seed require clean protocol-v1 audits from the frozen experiment commit, the
    generator/executor reject stale seed evidence, and repeated macro protocol v2
    refuses completion unless all 18 cells retain valid auxiliary wall-time
    accounting;
11. freeze adaptive plan protocol v3 after all refined primary scored artifacts
    have clean frozen-commit audits, then run RQ3 base-budget generations and all
    frozen 25/50/75% cap analyses with scripts/execute_adaptive_plan.py; refined
    inputs and reusable base raw/scored artifacts must remain protocol-v1,
    clean, commit-matched, stage-correct, and SHA-bound;
12. optionally run FLARE same-checkpoint RQ4 after its parity/trace gates;
13. render paper numbers only through scripts/render_results_tex.py; the renderer
    re-evaluates the frozen RQ1 gate from configs/experiments/primary.yaml and
    SHA-binds the primary, claim-gate, repeated, feature-ablation, adaptive,
    and primary-config sources;
14. commit/stash the final paper-side changes so the submission checkout is clean,
    then run scripts/submission_readiness.py. The submission gate revalidates the
    historical experiment artifacts without requiring the paper commit to equal
    the experiment commit, while separately requiring the current submission
    checkout to be clean. It also re-checks source hashes, anonymity, and the
    deterministic software supplement before ACL pubcheck/upload.

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
- #8 - frozen 18-cell primary matrix, primary analysis, and feature-ablation macro
- #9 - RQ2 five-sample comparator with compute accounting
- #10 - RQ3 adaptive 25/50/75% matrix
- #11 - artifact-derived paper rendering and final ARR submission readiness

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

The renderer checks complete cell sets, re-evaluates the RQ1 claim gate from the
frozen primary config, and writes a manifest that SHA-binds five result
artifacts (primary, RQ1 gate, repeated sampling, feature ablations, and adaptive
compute) plus the primary config before populating paper macros. Final
submission readiness then distinguishes the historical experiment commit from
the later clean paper/submission commit.

## Execution tracking issues

The repository now tracks every remaining runtime stage explicitly:

- #1 LLaDA + GSM8K 50-example pilot and human trace review;
- #2 SDAR exact parity;
- #3 optional FLARE/HybridDiffusion RQ4 diagnostic;
- #4 MATH-500 automatic/manual scorer gate;
- #5 Dream exact parity;
- #6 hosted CI runner-allocation infrastructure;
- #7 LLaDA-MoE exact parity;
- #8 frozen 18-cell primary matrix plus RQ1 analysis;
- #9 RQ2 five-sample comparator with complete auxiliary compute accounting;
- #10 RQ3 adaptive second-pass matrix at the frozen 25/50/75% caps;
- #11 artifact-derived paper rendering and final ARR submission readiness.

Issues #1, #2, #4, #5, and #7 remain open until real A100 artifacts pass.
Issues #8--#11 are execution checklists, not evidence that those experiments are
already complete.

## Commands to run first on a fresh continuation

~~~bash
git pull
python scripts/project_status.py
~~~

Then freeze the pre-primary provenance before GPU work.

Windows:

~~~powershell
PowerShell -ExecutionPolicy Bypass -File scripts/freeze_preprimary_windows.ps1
~~~

Linux/macOS:

~~~bash
bash scripts/freeze_preprimary.sh
~~~

For the NVIDIA A100 80GB server, follow `docs/A100_EXECUTION.md` as the
canonical copy/paste path. Both freeze wrappers enforce one clean experiment
commit and export the private pre-primary handoff. Then follow
docs/PILOT_RUNBOOK.md for detailed first-gate rationale and
docs/PRIMARY_MATRIX_RUNBOOK.md after readiness passes.

## New-chat continuation prompt

A future chat can start with:

> Continue the TrajUQ NAACL 2027 project from the repository
> Rocky5502/TrajUQ. Read PROJECT_STATUS.md, configs/project_status.yaml, the
> open GitHub issues, and the latest main branch before changing anything.
> Preserve the frozen 3x6 primary design, outcome isolation, complete 18-cell
> feature-ablation macro, RQ1 claim gate, five-sample 18-cell comparator, and
> RQ3 50% primary cap with 25/75% sensitivities. Do not invent results. Continue from the first unmet execution
> gate and update PROJECT_STATUS.md whenever a milestone materially changes.

## Maintenance rule

Whenever a milestone materially changes, update all three together:

1. the relevant GitHub issue;
2. configs/project_status.yaml if the expected artifact/protocol changes;
3. this PROJECT_STATUS.md handoff file.

This makes the repository - not the chat transcript - the canonical project
memory.
