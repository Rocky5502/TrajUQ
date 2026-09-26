# Primary matrix runbook

The full primary matrix is intentionally impossible to plan until the frozen
experiment commit, integration gates, scorer gates, and readiness gate all pass.
The order below is part of the reproducibility protocol.

## 1. Freeze one clean experiment commit, CPU verification, and immutable revisions

Do this **before** model parity. The same git commit must anchor CPU verification,
the revision ledger, and all three primary-model parity artifacts.

Windows preferred path:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/freeze_preprimary_windows.ps1
```

Linux/macOS equivalent:

```bash
bash scripts/freeze_preprimary.sh
```

For the NVIDIA A100 80GB Linux path, follow
`docs/A100_EXECUTION.md` end-to-end rather than mixing low-level commands from
different runbooks.

Do not change tracked experiment code after the freeze passes. The freeze
wrapper runs the CPU verifier and revision resolver at one clean commit. The
revision resolver requires Hugging Face access to all frozen primary assets,
including GPQA. It writes:

- `results/revision_ledger.json` with protocol
  `trajuq_revision_ledger_v2`;
- the experiment git commit;
- hashes of the primary config and dataset manifest;
- immutable model and dataset SHAs;
- hash-bound model configs under `results/frozen_models/`.

If any repository cannot be resolved, stop rather than substituting a moving
branch or a different benchmark source.

## 2. Pass the pre-primary execution gates at that frozen commit

Required execution evidence includes:

- `results/cpu_verification.json` = clean PASS at the ledger git commit;
- LLaDA exact parity v3 = PASS on exactly 20 shared prompts;
- LLaDA + GSM8K 50-example calibration pilot quality = PASS;
- human review of at least 10 pilot trajectories = PASS;
- SDAR exact parity v3 = PASS on exactly 20 shared prompts;
- Dream exact parity v3 = PASS on exactly 20 shared prompts;
- automatic MATH-500 symbolic scorer audit = PASS;
- calibration-only MATH-500 manual scorer review of at least 25 examples = PASS.

Every parity-v3 artifact must match the revision-ledger git commit and the
ledger-resolved model SHA. Do not regenerate the ledger after seeing parity or
pilot outcomes merely to make an artifact pass.

For the first LLaDA path on Windows:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage prepare
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage parity
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage pilot
```

Then complete the required human pilot attestation, SDAR/Dream parity protocols,
and MATH-500 scorer review using their dedicated runbooks.

## 3. Run the primary readiness hard gate

Only after the preceding artifacts exist:

```bash
python scripts/primary_readiness.py
```

Readiness protocol v2 rejects stale or mixed provenance. In particular it
requires the clean CPU artifact and every parity artifact to share the
revision-ledger git commit, requires each parity model revision to equal the
ledger SHA, and requires exactly 20 shared parity prompts.

Do not continue unless `results/primary_readiness.json` says `PASS`.

## 4. Prepare all dataset snapshots

Use the exact dataset SHAs from `results/revision_ledger.json` and prepare each
pair:

```bash
python scripts/prepare_eval_pair.py \
  --dataset <DATASET> \
  --revision <EXACT_DATASET_SHA>
```

The planner will reject a split manifest whose `requested_revision` is empty.
GPQA remains gated and must be accessed under its upstream terms.

## 5. Build, inspect, and archive the execution plan

```bash
python scripts/plan_primary_matrix.py
```

For the frozen 3-model × 6-dataset primary configuration, the plan contains 18
model--dataset cells and separate calibration/test generation, raw audit,
scoring, and scored-audit argv arrays for each cell.

The planner **does not execute GPU work**. This is deliberate: inspect and archive
`results/primary_matrix_plan.json` together with the freeze manifests before
launching expensive runs.

## 6. Execute without mutation

The preferred resumable executor is:

```bash
# Dry-run/status first:
python scripts/execute_primary_plan.py

# Execute only one cell:
python scripts/execute_primary_plan.py --cell llada:gsm8k --execute

# Resume the complete frozen plan serially:
python scripts/execute_primary_plan.py --execute
```

The executor never overwrites existing raw/scored artifacts. It reuses an
existing artifact only when the corresponding audit is protocol
`trajuq_run_audit_v1`, was recorded from a clean checkout at the frozen
experiment commit, has the correct raw/scored stage, and its stored SHA-256
matches the current file. After every executed step it rewrites
`results/primary_execution_state.json` and appends an event to
`results/primary_execution_events.jsonl`, making interruption/restart status
explicit.

When executing the plan:

- use the argv values exactly;
- never overwrite completed raw runs;
- never regenerate data partitions after outcomes are seen;
- do not edit the feature dictionary, calibration family, or statistical rules
  because a test result is weak;
- preserve raw/scored artifacts separately;
- archive every audit, environment snapshot, and SHA-256 manifest.

After cell-level metrics are exported, use `scripts/aggregate_cells.py` for the
hierarchical equal-cell-weight macro analysis.


## 7. Adaptive second-pass plan

The full-budget primary outputs double as the refined-budget artifacts for RQ3.
After all 18 refined primary scored artifacts and their audits exist, generate
the separate low-budget base-run plan:

```bash
python scripts/plan_adaptive_runs.py
```

The frozen budgets are:

| Model | Base | Refined |
|---|---:|---:|
| LLaDA | 32 steps | 128 steps |
| Dream | 32 steps | 128 steps |
| SDAR | 2 denoising steps/block | 4 denoising steps/block |

The primary selection cap is 50%; 25% and 75% are pre-registered sensitivity
settings. The v3 planner emits all three evaluation commands for every one of
the 18 model--dataset cells and exports per-example adaptive decisions. After
all cells are complete, run the plan's `aggregate_adaptive.py` command for
`cap_050`, then the two sensitivity commands for `cap_025` and `cap_075`.
Each macro result uses equal cell weight and a paired within-cell hierarchical
bootstrap.

Adaptive plan protocol v3 SHA-binds the frozen primary plan/config, every
calibration/test prompt input, every refined scored artifact, and each scored
audit. Refined scored audits must be clean protocol-v1 evidence from the frozen
experiment commit. `execute_adaptive_plan.py` revalidates those bindings before
any low-budget GPU work, and reusable base raw/scored artifacts must satisfy the
same protocol/clean/commit/stage/SHA contract, so stale evidence cannot silently
enter RQ3.

Selected examples use the conservative **second-pass rerun** semantics: the base
run is always paid and the refined run is an additional complete generation.
The primary macro report includes the adaptive accuracy delta versus base,
fully refined, and expected random second-pass selection, plus a nominal cost
ratio relative to the base budget. The planner never uses the cheaper
continuation cost formula unless a future runtime genuinely resumes the same
denoising trajectory.


## 8. Freeze the primary analysis plan

After all 18 calibration/test pairs are generated, audited, and scored, create
the analysis command ledger:

```bash
python scripts/plan_primary_analysis.py
```

This writes `results/primary_analysis_plan.json`. Protocol v2 requires every
scored calibration/test artifact to have a clean protocol-v1 scored audit from
the frozen experiment commit, then SHA-binds those inputs/audits together with
the frozen primary plan/config and experiment commit. For every model--dataset cell it freezes commands for the calibrated
baseline/TrajUQ metrics, paired nested bootstrap comparison, frozen
feature-family ablations, and the training-free raw-risk diagnostic.

Only after **all 18** cell analyses exist should the plan's macro aggregation
command be executed. The macro command rejects selective cell omission. Run the
RQ1 claim-gate command last; a negative gate is a valid result and must not be
used as a reason to redesign the test analysis.

The preferred resumable analysis executor is:

```bash
# Dry-run/status:
python scripts/execute_primary_analysis.py

# Execute one cell without touching other cells:
python scripts/execute_primary_analysis.py --cell llada:gsm8k --execute

# Resume all missing cell analyses and, only when all 18 are complete,
# run the macro aggregation and frozen RQ1 claim gate:
python scripts/execute_primary_analysis.py --execute
```

The executor first revalidates the plan-bound primary plan/config, scored inputs,
and scored-audit hashes **plus** audit protocol, clean-checkout state, stage, and
frozen experiment commit, then refuses to overwrite partial or invalid result
pairs. It writes `results/primary_analysis_execution_state.json` and a
chronological event log so analysis progress remains recoverable after interruption.


## 9. Freeze the expensive repeated-sampling plan

After primary analysis has produced the complete 18-cell TrajUQ prediction
artifacts, create the five-sample baseline ledger:

```bash
python scripts/plan_repeated_sampling.py
```

The frozen comparator now covers all 18 cells. Repeated-sampling plan protocol
v2 SHA-binds the primary plan, primary-analysis plan, primary config, prompt
partitions, scored targets/audits, and per-cell TrajUQ prediction artifacts before
auxiliary generation. Every bound scored target must have a clean protocol-v1
scored audit from the frozen experiment commit. MATH-500 sample votes use reference-free pairwise
Math-Verify equivalence clustering instead of string matching. The plan fixes five seeds
(2027--2031), temperature 0.2, model revision, prompt partitions, and all
adapter-specific generation settings.

Execute every planned auxiliary generation and cell evaluation before running
the plan's `aggregate_repeated_baseline.py` command. Repeated macro protocol v2
requires complete test-time auxiliary wall-clock metadata in all 18 cells and
reports the equal-cell-weight mean auxiliary seconds per example, excluding the
target generation itself. The executor and paper renderer both reject a macro
that drops this compute accounting. This full-matrix comparator is reported
separately because it is an expensive secondary baseline rather than the
pre-registered RQ1 endpoint.


## 10. Execute the five-sample comparator resumably

After the repeated-sampling plan is frozen, use:

```bash
# Dry-run/status:
python scripts/execute_repeated_sampling.py

# Execute one model--dataset cell:
python scripts/execute_repeated_sampling.py --cell llada:gsm8k --execute

# Resume the complete 18-cell comparator:
python scripts/execute_repeated_sampling.py --execute
```

Before any five-seed generation, the executor revalidates every upstream
SHA-bound artifact recorded by repeated-sampling plan protocol v2. Each
five-seed generation command then uses `--resume-verified`: an existing seed is
reused only when its raw audit is protocol-v1, clean, from the frozen experiment
commit, stage=`raw`, and SHA-bound to the current artifact. The repeated
manifest also records the git commit/clean state and input SHA-256. Unverified
partial artifacts are never overwritten automatically. The executor writes
`results/repeated_sampling_execution_state.json` and an append-only event log.
The full macro is run only after all 18 cells are complete, and the execution
state marks it complete only for protocol `trajuq_vs_repeated_macro_v2` with
complete five-sample auxiliary compute accounting.


## 11. Execute the adaptive RQ3 matrix resumably

After `results/adaptive_matrix_plan.json` is frozen, use:

```bash
# Dry-run/status:
python scripts/execute_adaptive_plan.py

# One cell, all frozen caps:
python scripts/execute_adaptive_plan.py --cell llada:gsm8k --execute

# Only the primary 50% cap:
python scripts/execute_adaptive_plan.py --fraction-key cap_050 --execute

# Resume the complete 18-cell adaptive matrix:
python scripts/execute_adaptive_plan.py --execute
```

The executor first creates/audits/scores the low-budget base artifacts, then
runs the frozen 25%, 50%, and 75% cap evaluations. Existing partial result
pairs are never overwritten automatically. Hierarchical macro aggregation for a
cap is blocked until all 18 cells for that cap are complete. Progress is written
to `results/adaptive_execution_state.json` with an append-only event log.
