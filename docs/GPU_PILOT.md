# Local GPU pre-primary execution

This document is the current **Windows/local-GPU** entry point for the first real TrajUQ experiments.

The scientific design is already frozen. Do **not** change the model family, dataset suite, generation settings, feature set, or statistical gate to make a local run fit the machine. If the frozen checkpoint does not fit or parity fails, stop and move that gate to the A100/Linux path rather than silently substituting another model or precision policy.

## 0. Pull the exact repository state

From the TrajUQ repository:

```powershell
git pull
git status --short
python scripts/project_status.py
```

The worktree must be clean before freezing. Existing immutable `results/*` freeze evidence from another commit must be inspected/archived rather than mixed with a new freeze.

## 1. Freeze the clean experiment commit

Run:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/freeze_preprimary_windows.ps1
```

This is the required first gate. It:

- runs the complete CPU verification suite;
- resolves immutable Hugging Face model and dataset SHAs;
- verifies access to gated primary datasets (currently GPQA-Diamond);
- exercises exact-revision normalizer/scorer contracts on all six primary datasets;
- validates that all evidence is bound to the same clean git commit;
- refreshes the machine-readable project status;
- exports `dist/trajuq_preprimary_handoff.zip`.

Do not start GPU parity unless all of these exist and PASS:

- `results/cpu_verification.json`
- `results/revision_ledger.json`
- `results/gated_dataset_access.json`
- `results/dataset_contracts.json`

The frozen model configs are written under `results/frozen_models/`.

## 2. Start the first real GPU experiment: LLaDA parity

Prepare the exact-revision GSM8K pair and run the CUDA preflight:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage prepare
```

Then run the 20-prompt native-vs-instrumented parity gate:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage parity
```

Required output:

`results/llada_parity.json`

The gate must be protocol `trajuq_exact_parity_v3` with:

- exactly 20 shared prompts;
- zero token/output mismatches;
- zero provenance mismatches;
- the ledger-resolved LLaDA model SHA;
- the frozen experiment git commit;
- clean source-checkout provenance;
- matching native/instrumented runtime provenance.

If parity fails, **stop**. Debug the adapter/runtime mismatch before any 50-example pilot or primary run.

## 3. Run the 50-example LLaDA + GSM8K pilot

Only after exact LLaDA parity PASS:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage pilot
```

This produces and audits:

- `runs/llada_gsm8k_pilot50.raw.jsonl`
- `runs/llada_gsm8k_pilot50.scored.jsonl`
- `results/llada_gsm8k_pilot50_gate.json`
- `results/llada_gsm8k_pilot50_trace_review.md`

The automated pilot gate checks record count, outcome isolation, model revision, telemetry semantics, feature variation, timing/memory metadata, and both correctness classes.

## 4. Human pilot review — mandatory stop

Inspect at least the 10 exported trajectories in:

`results/llada_gsm8k_pilot50_trace_review.md`

Check the generated text, step summaries, entropy/JS/revision behavior, active-position semantics, commitment behavior, and absence of outcome/reference leakage.

Only after actual inspection, record the human attestation with `scripts/record_manual_gate.py` using the review sheet and the required confirmation flags. Do not create this artifact automatically.

Expected artifact:

`results/llada_gsm8k_pilot50_manual_review.json`

## 5. Remaining primary-model parity

After the LLaDA pilot/manual gate is healthy, run:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/parity_remaining_windows.ps1 -Model sdar
PowerShell -ExecutionPolicy Bypass -File scripts/parity_remaining_windows.ps1 -Model dream
```

Both are exact 20-prompt parity-v3 gates at the same frozen commit and their ledger-resolved model SHAs.

Expected artifacts:

- `results/sdar_parity.json`
- `results/dream_parity.json`

## 6. MATH-500 scorer gate

Prepare/revalidate the frozen MATH-500 scorer setup and run the 25-example calibration-only scorer pilot:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/math500_gate_windows.ps1 -Stage all
```

Then manually inspect all 25 rows in:

`results/math500_scorer_review.md`

Only after that review, record the MATH-500 human scorer attestation with `scripts/record_math500_scorer_review.py`.

Expected final artifact:

`results/math500_scorer_manual_review.json`

Never hand-flip final test labels.

## 7. Primary readiness

When all pre-primary gates above are complete:

```powershell
python scripts/primary_readiness.py
```

Require:

- protocol `trajuq_primary_readiness_v2`;
- `status=PASS`;
- current worktree clean and equal to the frozen experiment commit;
- all three parity artifacts source- and runtime-bound;
- pilot and human-review artifacts hash-bound;
- MATH-500 automatic/manual scorer gates PASS;
- frozen dataset/config/revision evidence unchanged.

Do **not** launch the 18-cell primary matrix before this gate passes.

## Local GPU versus A100

The local GPU is appropriate for the **first pre-primary gates** if the frozen checkpoints load successfully and CUDA preflight passes. No scientific setting should be weakened to force them to fit.

For the full 18-cell primary matrix, five-sample comparator, and adaptive-compute campaign, prefer the guarded A100/Linux runner in:

- `scripts/a100_campaign_linux.sh`
- `docs/A100_EXECUTION.md`

If a local run hits OOM or runtime instability, preserve the artifacts/logs, stop, and move the same frozen commit/revisions to A100. Do not substitute a smaller checkpoint and call it the preregistered experiment.

## Progress handoff

Because raw runs and most `results/*` artifacts are intentionally gitignored, package compact evidence after meaningful milestones:

```powershell
python scripts/project_status.py
python scripts/export_research_handoff.py --output dist/trajuq_local_progress_handoff.zip --skip-status-refresh
```

Upload that ZIP in the next chat when you want the local evidence inspected. It excludes raw run JSONL, processed benchmark data, model weights, and known secret patterns.
