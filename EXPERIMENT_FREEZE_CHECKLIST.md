# Experiment freeze checklist

For the repository-wide continuation state and unmet execution gates, read
`PROJECT_STATUS.md` first and run `python scripts/project_status.py`.

Do not start the full primary matrix until every required item is checked.

## Scientific definition
- [ ] RQ1–RQ4 wording frozen.
- [ ] Primary model families frozen.
- [ ] Primary datasets/splits frozen.
- [ ] Calibration/test partition files frozen before model calls.
- [ ] Primary trajectory feature vector frozen.
- [ ] Transparent raw-risk equation/version frozen.
- [ ] Final-state and repeated-generation baselines frozen.
- [ ] Calibration method/split frozen.
- [ ] Primary metrics frozen.
- [ ] Paired bootstrap protocol and seed frozen.
- [ ] Adaptive policy and matched-compute comparison frozen.

## Model/runtime integrity
- [ ] On a rented A100 server, `results/a100_rental_clock.json` is created immediately after checking out the exact commit, before dependency installation/downloads, and binds the same clean commit to the frozen 72-hour budget.
- [ ] One clean experiment git commit selected before GPU parity.
- [ ] `results/cpu_verification.json` is PASS on that exact commit.
- [ ] `results/revision_ledger.json` is protocol `trajuq_revision_ledger_v2` and PASS on that exact commit.
- [ ] `results/gated_dataset_access.json` is protocol `trajuq_gated_dataset_access_v1`, PASS on that same clean commit, and SHA-bound to the revision ledger.
- [ ] Every gated primary dataset (currently GPQA-Diamond) can stream one record at its ledger-resolved SHA without storing benchmark content.
- [ ] `results/dataset_contracts.json` is protocol `trajuq_dataset_contracts_v1`, PASS on that same clean commit, and SHA-bound to the revision ledger.
- [ ] All six ledger-pinned primary datasets pass the live normalizer/scorer smoke, with no streamed benchmark row content stored in the contract artifact.
- [ ] Primary config and dataset-manifest hashes match the revision ledger.
- [ ] Exact model/tokenizer and dataset SHAs are recorded in the revision ledger.
- [ ] Frozen model-config hashes match the revision ledger.
- [ ] Exact upstream runtime commit recorded where applicable.
- [ ] Mask ID / special-token assumptions audited.
- [ ] LLaDA, SDAR, and Dream parity artifacts are protocol `trajuq_exact_parity_v3`.
- [ ] Every primary parity artifact covers exactly 20 shared prompts with zero output/provenance mismatches.
- [ ] Every primary parity artifact matches the revision-ledger git commit and model SHA.
- [ ] Pilot output matches the exact state whose trajectory was logged.
- [ ] `results/a100_machine_preflight.json` is protocol `trajuq_a100_machine_preflight_v1` and PASS on the frozen commit: exactly one visible A100, >=75 GiB VRAM, bfloat16 support, >=40 GiB host RAM, >=80 GiB free disk, healthy `nvidia-smi`.
- [ ] Frozen model configs require `device: cuda` and `dtype: bfloat16`; the A100 reports CUDA available and bfloat16 support.
- [ ] Precision, CUDA, PyTorch, Transformers, and GPU recorded.

## Engineering
- [ ] 50-item pilot completes on one model.
- [ ] Pilot raw and scored `audit_run.py` artifacts are PASS and still SHA-bind their files.
- [ ] At least 10 trajectory records manually inspected.
- [ ] Pilot review sheet identifies the exact scored-pilot path/SHA reviewed.
- [ ] Human pilot attestation binds that review sheet, scored source, parity artifact, and pilot-quality artifact.
- [ ] Raw-run audit PASS.
- [ ] No NaN/Inf in features.
- [ ] Duplicate IDs rejected.
- [ ] Resume logic verified.
- [ ] Timing and peak GPU memory recorded.
- [ ] Output SHA-256 recorded.
- [ ] Unit tests pass at the exact experiment commit.

## Outcome isolation
- [ ] Raw generation has `correct=null`.
- [ ] Raw generation has no reference answer.
- [ ] Correctness is joined only after raw audit.
- [ ] Test labels are never used for feature/calibrator/threshold selection.
- [ ] Scored artifact is separate and immutable.

## Data / release integrity
- [ ] GPQA terms accepted locally; no gated rows committed.
- [ ] Dataset revisions/fingerprints recorded.
- [ ] No API/HF tokens or private paths committed.
- [ ] No third-party model weights committed.
- [ ] MATH-500 symbolic verifier frozen before MATH-500 primary scoring.
- [ ] Every paper number maps to a frozen result artifact and code commit.


## Pre-primary automation artifacts
- [ ] `results/cpu_verification.json` reports clean PASS.
- [ ] `results/revision_ledger.json` reports protocol-v2 PASS.
- [ ] `results/gated_dataset_access.json` reports protocol-v1 PASS and binds the current revision-ledger SHA-256.
- [ ] `results/dataset_contracts.json` reports protocol-v1 PASS for all six primary datasets and binds the current revision-ledger SHA-256.
- [ ] `results/llada_parity.json` reports protocol-v3 PASS for exactly 20 prompts.
- [ ] `results/sdar_parity.json` reports protocol-v3 PASS for exactly 20 prompts.
- [ ] `results/dream_parity.json` reports protocol-v3 PASS for exactly 20 prompts.
- [ ] `results/llada_gsm8k_pilot50_gate.json` reports PASS for exactly 50 raw/scored records.
- [ ] Pilot raw/scored `.audit.json` files report PASS and match the pilot artifacts by SHA-256.
- [ ] `results/llada_gsm8k_pilot50_trace_review.md` manually reviewed and source-bound to that scored pilot.
- [ ] At least 10 trajectory traces checked for logging/parsing plausibility.
- [ ] `results/math500_scorer_audit.json` reports clean protocol-v2 PASS with frozen dependency versions.
- [ ] MATH-500 scorer review sheet is source-bound to the frozen calibration input and scored 25-example pilot.
- [ ] MATH-500 scored-pilot `.audit.json` reports PASS and matches the reviewed scored artifact.
- [ ] `results/math500_scorer_manual_review.json` reports PASS after >=25 calibration-only examples.
- [ ] `results/primary_readiness.json` reports protocol `trajuq_primary_readiness_v2` and PASS.
