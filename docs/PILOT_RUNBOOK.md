# First GPU pilot runbook

## Windows frozen-commit + staged wrapper

Before any GPU parity run, freeze one clean experiment commit:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/freeze_preprimary_windows.ps1
```

This installs the CPU/model/data dependencies unless skipped, runs the complete
CPU verification suite, resolves immutable revisions for all three primary
models and all six datasets, and writes `results/revision_ledger.json` plus
`results/frozen_models/*.yaml`. GPQA is gated, so Hugging Face authentication
must already permit access.

Do not edit tracked experiment code after this freeze. Then run:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage prepare
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage parity
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage pilot
```

The pilot wrapper consumes the ledger-resolved LLaDA and GSM8K SHAs. It is
non-overwriting: valid hash-bound frozen evidence is preserved, while partial or
stale artifacts stop for inspection. It hard-stops unless parity protocol v3
covers exactly 20 shared prompts with zero output/provenance mismatches at the
same clean git commit/model SHA as the ledger, with unchanged reference and
instrumented SHA-256 evidence.

The first pilot is frozen to the **LLaDA-MoE-7B-A1B-Instruct + GSM8K calibration** path. Do not scale to other models until the exact token-parity gate passes.

## A. Freeze immutable revisions at one experiment commit

Preferred Windows command:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/freeze_preprimary_windows.ps1
```

Linux/macOS guarded equivalent:

```bash
bash scripts/freeze_preprimary.sh
```

On a Linux GPU server (including an NVIDIA A100), the guarded LLaDA path is:

```bash
bash scripts/pilot_linux.sh prepare
bash scripts/pilot_linux.sh parity
bash scripts/pilot_linux.sh pilot
```

The Linux wrapper mirrors the Windows evidence contract: it requires the same
clean CPU/revision-ledger commit, preserves existing valid evidence, refuses to
overwrite partial/stale artifacts, revalidates the frozen GSM8K split and pilot
manifest, requires exact parity-v3 with 20 shared prompts, and only then runs the
50-example pilot. Use Python >=3.10; Ubuntu 20.04's system Python may be older.

Recommended A100/Ubuntu sequence after the repository code is frozen:

```bash
nvidia-smi
python --version
bash scripts/freeze_preprimary.sh
bash scripts/pilot_linux.sh prepare
bash scripts/pilot_linux.sh parity
bash scripts/pilot_linux.sh pilot
```

After manually reviewing the exported ten LLaDA/GSM8K trajectories, record the
attestation:

```bash
python scripts/record_manual_gate.py \
  --review-sheet results/llada_gsm8k_pilot50_trace_review.md \
  --reviewed-records 10 \
  --trajectory-semantics-pass \
  --scorer-audit-pass \
  --parity-artifact-reviewed \
  --no-leakage-observed \
  --notes "Reviewed frozen 10-example pilot sheet on the A100 run."
```

Then clear the other two primary-model parity gates on the same frozen checkout:

```bash
bash scripts/parity_remaining_linux.sh all
```

Run the frozen MATH-500 scorer path:

```bash
bash scripts/math500_gate_linux.sh prepare
bash scripts/math500_gate_linux.sh pilot
```

After manually reviewing all 25 exported MATH-500 calibration examples, record
that attestation:

```bash
python scripts/record_math500_scorer_review.py \
  --review-sheet results/math500_scorer_review.md \
  --reviewed-records 25 \
  --answer-extraction-pass \
  --symbolic-equivalence-pass \
  --no-manual-label-flips \
  --notes "Reviewed frozen 25-example MATH-500 scorer sheet."
```

Finally:

```bash
python scripts/primary_readiness.py
```

Only a protocol-v2 `status=PASS` result authorizes creation/execution of the
full 18-cell primary matrix.

This performs the same clean-checkout CPU verification, immutable revision
resolution, post-freeze cleanliness check, status refresh, and private handoff
export as the Windows wrapper. Revision-ledger protocol v2
records the git commit, primary-config hash, dataset-manifest hash, immutable
model/data SHAs, and frozen model-config hashes. Primary readiness v2 later
requires CPU verification and every primary parity artifact to match this
provenance.

## B. Prepare the frozen calibration/test pair

```bash
python scripts/prepare_eval_pair.py \
  --dataset gsm8k \
  --revision <EXACT_GSM8K_SHA>
```

Do not regenerate these files after seeing model outcomes.

## C. Preflight and manifest

```bash
python scripts/preflight.py \
  --model-config results/frozen_models/llada.yaml \
  --data data/processed/gsm8k/calibration.jsonl \
  --require-cuda

python scripts/environment_snapshot.py --output results/pilot_environment.json

python scripts/freeze_manifest.py \
  --model-config results/frozen_models/llada.yaml \
  --dataset data/processed/gsm8k/calibration.jsonl \
  --dataset data/processed/gsm8k/test.jsonl \
  --output results/pilot_freeze_manifest.json
```

## D. Twenty-prompt hard parity gate

Run the native uninstrumented reference:

```bash
python scripts/run_llada_reference.py \
  --model inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --revision <EXACT_MODEL_SHA> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/llada_parity_reference.jsonl \
  --limit 20 \
  --steps 128 \
  --max-new-tokens 128 \
  --block-length 128 \
  --temperature 0
```

Run TrajUQ on the same prompts:

```bash
python scripts/run_generation.py \
  --adapter llada \
  --model inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --revision <EXACT_MODEL_SHA> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/llada_parity_trajuq.raw.jsonl \
  --limit 20 \
  --steps 128 \
  --max-new-tokens 128 \
  --block-length 128
```

Audit and compare:

```bash
python scripts/audit_run.py \
  --run runs/llada_parity_trajuq.raw.jsonl \
  --stage raw \
  --require-revision

python scripts/compare_run_parity.py \
  --reference runs/llada_parity_reference.jsonl \
  --instrumented runs/llada_parity_trajuq.raw.jsonl \
  --require-token-ids \
  --require-provenance \
  --output results/llada_parity.json
```

**STOP if parity is not PASS.** Do not generate primary/pilot paper numbers until the adapter is repaired.

## E. Fifty-example pilot

After parity passes, use a fresh output file:

```bash
python scripts/run_generation.py \
  --adapter llada \
  --model inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --revision <EXACT_MODEL_SHA> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/llada_gsm8k_pilot50.raw.jsonl \
  --limit 50 \
  --steps 128 \
  --max-new-tokens 128 \
  --block-length 128
```

Then:

```bash
python scripts/audit_run.py \
  --run runs/llada_gsm8k_pilot50.raw.jsonl \
  --stage raw \
  --require-revision

python scripts/score_outputs.py \
  --run runs/llada_gsm8k_pilot50.raw.jsonl \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/llada_gsm8k_pilot50.scored.jsonl
```

Manually inspect at least ten trajectories before interpreting any UQ metric.

## Pilot pass criteria

Proceed to the full matrix only if:
- exact 20-prompt parity is PASS;
- raw and scored audits are PASS and still SHA-bind their corresponding files;
- all features are finite;
- trajectory features show non-trivial variance;
- scorer output is manually plausible;
- no prompt/reference leakage is present;
- no repeated OOM/runtime instability exists;
- timing and peak GPU memory are present;
- the frozen code/data/model revisions are recorded.

A weak uncertainty signal is not a reason to redesign features using final test labels. It is a scientific result or a calibration-set design decision.


## Automated Windows wrapper

Run the frozen-commit setup once, then the LLaDA stages separately on the first
pass:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/freeze_preprimary_windows.ps1
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage prepare
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage parity
PowerShell -ExecutionPolicy Bypass -File scripts/pilot_windows.ps1 -Stage pilot
```

The freeze wrapper records one clean experiment commit, runs CPU verification,
resolves all primary model/data SHAs, and writes hash-bound frozen model configs.
The pilot wrapper prepares the frozen GSM8K pair, snapshots the environment,
runs 20-prompt native/instrumented parity from those exact revisions, measures
telemetry overhead, hard-stops on mismatch, runs the 50-example pilot, audits
raw/scored artifacts, checks feature variation/timing/schema integrity, and
exports a 10-example manual review sheet. Before every parity/pilot stage it
revalidates the frozen split/audit, frozen LLaDA config, and pilot-freeze
manifest hashes. Existing pilot evidence is reused only if both raw/scored
`audit_run.py` artifacts are still PASS and SHA-bound.

The automated gate does not replace the required human inspection of at least
ten trajectories. The pilot quality artifact is protocol v2 and binds its raw
and scored files by SHA-256. The review sheet additionally records the exact
scored-pilot path and SHA-256 reviewed by the human.

## Reproducibility note

The tracked model YAML remains a template with `revision: null`. The immutable revisions used for a concrete run are stored in the v2 revision ledger and hash-bound frozen model configs, preventing a pilot run from silently dirtying the repository's generic configuration or re-resolving a moving model head.


## G. Record the human review

After actually inspecting the exported 10-record sheet and scorer behavior:

```bash
python scripts/record_manual_gate.py \
  --review-sheet results/llada_gsm8k_pilot50_trace_review.md \
  --reviewed-records 10 \
  --trajectory-semantics-pass \
  --scorer-audit-pass \
  --parity-artifact-reviewed \
  --no-leakage-observed \
  --notes "Manual review notes here."
```

The command is an attestation recorder, not an automated review. Do not run it
until those checks have actually been performed. It refuses a dirty worktree.
Before writing PASS it revalidates the parity source hashes, pilot-quality
scored source, and pilot raw/scored audits. Manual-gate protocol v2 binds the
review sheet, its exact scored-pilot source, LLaDA parity artifact, and
pilot-quality artifact by SHA-256 at the frozen experiment commit.
`primary_readiness.py` independently performs the same source/audit checks, so
editing an attestation JSON cannot bypass the evidence gate.

Before launching the full 3-model primary matrix, require all primary-model
parity artifacts plus the pilot gates:

```bash
python scripts/primary_readiness.py
```

The full matrix remains blocked unless that command returns `status=PASS`.
