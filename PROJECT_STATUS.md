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
- five-seed repeated-sampling baseline and full-matrix macro aggregation; repeated-sampling plan protocol v2 binds the primary/analysis plans, config, scored targets, prompt partitions, and TrajUQ prediction artifacts by SHA-256 before auxiliary generation;
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
- PowerShell and Bash CPU-verification entry points plus a one-command Windows
  pre-primary freeze wrapper;
- a guarded Windows parity wrapper for SDAR/Dream that consumes the frozen
  revision ledger, enforces a clean matching checkout, preserves valid PASS
  artifacts, and refuses ambiguous partial evidence;
- clean-checkout provenance guards across the pilot wrapper, primary readiness,
  all frozen planners, and all primary/repeated/adaptive executors, so an
  uncommitted source edit cannot masquerade under the frozen git SHA;
- primary-readiness v2 additionally revalidates the revision-ledger dataset
  manifest hash, the exact CPU-verification protocol/step sequence, frozen
  MATH-500 dependency versions, and clean-checkout human attestations;
- a versioned `NEW_CHAT_MASTER_PROMPT.md` plus a fail-closed private research
  handoff exporter that refuses to omit required continuation context.

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
   binds raw/scored files by SHA-256 and the frozen git commit/model SHA. Inspect
   at least 10 exported trajectories and record the protocol-v2 human attestation
   from the same frozen clean checkout; it binds the review sheet, parity
   artifact, and pilot-quality artifact by SHA-256. See issue #1.
5. Run SDAR exact parity protocol v3 at the same frozen commit/revision. On
   Windows, use `scripts/parity_remaining_windows.ps1 -Model sdar`; it binds
   the run to the clean CPU/revision freeze and refuses ambiguous partial
   artifacts. See issue #2.
6. Run Dream exact parity protocol v3 at the same frozen commit/revision. On
   Windows, use `scripts/parity_remaining_windows.ps1 -Model dream` with the
   same frozen-state and non-overwrite safeguards. See issue #5.
7. Require the MATH-500 automatic scorer audit protocol v2 at the frozen commit,
   then a >=25-example calibration-only MATH-500 scoring pilot, deterministic
   review sheet, and protocol-v2 human attestation bound to the audit/sheet
   hashes. On Windows, `scripts/math500_gate_windows.ps1 -Stage all` performs
   the automated portion from the frozen clean state and deliberately stops
   before the human attestation. Record that attestation from the same frozen
   clean checkout; readiness also requires the audit to report
   math-verify=0.9.0 and antlr4-python3-runtime=4.13.2. See issue #4.
8. Run `python scripts/primary_readiness.py` and require protocol
   `trajuq_primary_readiness_v2` with `status=PASS`. It rejects stale/mixed
   CPU, parity, pilot, human-review, MATH-500 scorer/dependency, dataset-manifest,
   commit, and dirty-worktree state.

Do **not** launch the full 18-cell primary matrix before these gates pass.

### Primary / analysis execution

After readiness passes:

1. prepare all six immutable dataset pairs;
2. create and archive results/primary_matrix_plan.json;
3. dry-run scripts/execute_primary_plan.py, then execute cells from the frozen
   plan; the executor writes a resumable SHA-verified execution state;
4. execute all 18 calibration/test cells exactly as planned;
5. audit and score every cell without overwriting completed artifacts;
6. create results/primary_analysis_plan.json only after all 18 scored artifacts and their audits exist; analysis-plan protocol v2 SHA-binds those inputs, the primary plan, config, and frozen experiment commit;
7. dry-run scripts/execute_primary_analysis.py and then use it to resume all
   cell-level metrics, nested comparisons, ablations, and raw-risk diagnostics; the executor refuses upstream provenance drift before reading result values;
8. aggregate the complete 18-cell primary endpoint;
9. run the RQ1 claim gate **without changing it after seeing the result**;
10. freeze repeated-sampling plan protocol v2 after primary analysis predictions exist, then run the complete 18-cell five-sample comparator with scripts/execute_repeated_sampling.py; verified seed artifacts are resumable and the executor rejects drift in bound primary artifacts;
11. freeze adaptive plan protocol v3 after all refined primary scored artifacts exist, then run RQ3 base-budget generations and all frozen 25/50/75% cap analyses with scripts/execute_adaptive_plan.py; the executor rejects drift in refined inputs/audits;
12. optionally run FLARE same-checkpoint RQ4 after its parity/trace gates;
13. render paper numbers only through scripts/render_results_tex.py; the renderer
    re-evaluates the frozen RQ1 gate from configs/experiments/primary.yaml and
    SHA-binds the primary, claim-gate, repeated, adaptive, and primary-config
    sources;
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
frozen primary config, and writes a manifest that SHA-binds all four result
artifacts plus the primary config before populating paper macros. Final
submission readiness then distinguishes the historical experiment commit from
the later clean paper/submission commit.

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
bash scripts/verify_cpu.sh
python scripts/resolve_experiment_revisions.py
~~~

Do not change commits between those commands. Then follow
docs/PILOT_RUNBOOK.md for the first GPU gate and
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
