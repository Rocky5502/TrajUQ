# TrajUQ Engineering Progress Log

This file records repository-side engineering progress separately from empirical
experiment completion. A committed implementation or test does **not** imply that
a local/GPU execution gate has passed. Runtime gates are authoritative only when
their expected `results/*` artifacts exist and pass inspection.

## 2026-09-27

### Canonical scientific state

- Target remains NAACL 2027 via ARR.
- Primary design remains frozen at 3 model families x 6 datasets = 18 cells.
- No empirical paper number has been invented or promoted from repository code.
- The first required runtime milestone remains the clean pre-primary freeze:
  `results/cpu_verification.json` PASS,
  `results/revision_ledger.json` protocol-v2 PASS,
  `results/gated_dataset_access.json` protocol-v1 PASS, and
  `results/dataset_contracts.json` protocol-v1 PASS at the same clean commit.
  Both data preflights are bound to the exact revision-ledger SHA-256.

### Continuity and pre-primary hardening

- Official Windows/Linux pre-primary freeze wrappers now perform an atomic
  evidence preflight before any producer runs. Existing CPU verification,
  revision-ledger, gated-access, MATH-500 scorer-audit, CPU-log, or frozen-model
  evidence stops the freeze before mixed-commit artifacts can be created.
- CPU verification, revision resolution, gated-access verification, and the
  automatic MATH-500 scorer audit refuse overwrite by default; explicit
  overwrite is recovery-only and is never requested by the official freeze
  wrappers.
- Added and versioned `NEW_CHAT_MASTER_PROMPT.md`.
- Made the private research-handoff exporter fail closed when required context is
  missing.
- The strict freeze wrappers now export
  `dist/trajuq_preprimary_handoff.zip` so gitignored evidence can be carried
  safely across chats/machines.
- Pilot/parity wrappers reject a dirty worktree and require the CPU verification
  artifact and revision ledger to match the frozen HEAD.
- Primary readiness revalidates the dataset-manifest hash, CPU protocol/step
  completion, per-step CPU verification log SHA-256 values, pilot/MATH review
  bindings, model parity evidence, and clean frozen checkout. CPU step logs are
  also included in the readiness source-binding manifest, so later planning and
  execution detect supporting-log drift.

### Frozen execution provenance

- Added `trajuq_dataset_contracts_v1`: the freeze streams one row from each
  exact-revision primary dataset, runs the real frozen normalizer and a
  constructed-correct scorer smoke, records only schema/provenance/status, and
  stores no benchmark row content. Primary readiness and the A100 campaign
  require the complete six-dataset same-commit ledger-bound PASS.
- Primary planning now re-hashes every gitignored `results/frozen_models/*.yaml`
  file and requires both its path and SHA-256 to match revision-ledger v2. A
  post-freeze change to model generation settings therefore blocks the 18-cell
  plan even when the tracked git checkout itself remains clean.
- Primary readiness now emits a SHA-256 source-binding manifest covering every
  file it validated, including nested parity/pilot/MATH artifacts and audits.
  Both primary planning and primary execution re-hash those bindings, closing
  the readiness-to-plan/execution time-of-check/time-of-use gap.
- Primary, primary-analysis, repeated-sampling, and adaptive planners/executors
  reject dirty worktrees and mismatched commits.
- Provenance validation is now recursively chained across stages: primary-analysis
  planning/execution revalidates the frozen primary plan; RQ2 planning/execution
  revalidates the primary-analysis plan (which revalidates primary/readiness);
  and RQ3 planning/execution revalidates the primary/readiness chain. Hashing an
  unchanged outer plan JSON is therefore not enough to hide drift in nested
  gitignored authorization evidence.
- Run audits are SHA-bound and commit-bound; reusable outputs must remain tied to
  their source artifacts.
- Primary-analysis cell outputs and primary/ablation/prior-signal macro outputs
  are source-bound before reuse.
- The RQ1 claim gate is source-bound to the primary macro and primary config.

### RQ2 / RQ3 aggregate hardening

- Repeated-sampling macro protocol v2 now records clean source provenance for:
  the frozen repeated plan, every consumed scored test target, TrajUQ prediction,
  repeated metric artifact, and repeated prediction artifact.
- `execute_repeated_sampling.py` re-hashes those sources before accepting an
  existing macro as COMPLETE.
- Adaptive cap macros now record clean source provenance for the frozen adaptive
  plan, every consumed refined scored test target, cap-level result, and cap-level
  prediction artifact.
- `execute_adaptive_plan.py` re-hashes those sources before accepting an
  existing cap macro as COMPLETE.
- Regression tests mutate consumed sources after macro creation and require the
  macro state to become INVALID.

### Paper/submission evidence chain

- The canonical result-artifact map now matches the renderer's seven-source
  contract, including the complete 18-cell prior-aligned signal diagnostic
  macro (`trajuq_prior_aligned_signal_macro_v1`) used by Appendix
  Table `tab:prior-signal-diagnostics`.
- Artifact-derived paper rendering requires provenance for primary, RQ1 claim,
  repeated, feature-ablation, prior-signal, and adaptive aggregate sources.
- The renderer revalidates upstream source hashes and frozen-commit agreement
  before writing result macros.
- Final submission readiness independently revalidates the same source chain and
  deterministic paper rendering.
- Result figures are also provenance-gated before reuse.

### Hosted CI status

GitHub Actions remains infrastructure-blocked before executable runner steps.
The last explicitly inspected failure (run #203 at commit
`c303a8becd78096a7f105265aad91cab8f164b59`) reported zero Ubuntu runner
milliseconds and no executable job steps/logs. This is not evidence of Ruff,
pytest, scorer, or paper-audit failure. Issue #6 tracks the infrastructure
problem.

### Required next runtime action

After pulling the final clean engineering commit, freeze again. Any artifacts
created at an earlier commit are stale by design.

A100/Linux:

```bash
git pull
git status --short
SKIP_INSTALL=1 bash scripts/freeze_preprimary.sh
```

Windows:

```powershell
git pull
git status --short
PowerShell -ExecutionPolicy Bypass -File scripts/freeze_preprimary_windows.ps1
```

Do not proceed to LLaDA parity until the CPU verification and revision ledger
both pass at that exact clean commit.
