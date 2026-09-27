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
  `results/revision_ledger.json` protocol-v2 PASS, and
  `results/gated_dataset_access.json` protocol-v1 PASS at the same clean
  commit, with gated access bound to the exact revision-ledger SHA-256.

### Continuity and pre-primary hardening

- Added and versioned `NEW_CHAT_MASTER_PROMPT.md`.
- Made the private research-handoff exporter fail closed when required context is
  missing.
- The strict freeze wrappers now export
  `dist/trajuq_preprimary_handoff.zip` so gitignored evidence can be carried
  safely across chats/machines.
- Pilot/parity wrappers reject a dirty worktree and require the CPU verification
  artifact and revision ledger to match the frozen HEAD.
- Primary readiness revalidates the dataset-manifest hash, CPU protocol/step
  completion, pilot/MATH review bindings, model parity evidence, and clean frozen
  checkout.

### Frozen execution provenance

- Primary, primary-analysis, repeated-sampling, and adaptive planners/executors
  reject dirty worktrees and mismatched commits.
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
