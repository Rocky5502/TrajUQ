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
- [ ] One clean experiment git commit selected before GPU parity.
- [ ] `results/cpu_verification.json` is PASS on that exact commit.
- [ ] `results/revision_ledger.json` is protocol `trajuq_revision_ledger_v2` and PASS on that exact commit.
- [ ] Primary config and dataset-manifest hashes match the revision ledger.
- [ ] Exact model/tokenizer and dataset SHAs are recorded in the revision ledger.
- [ ] Frozen model-config hashes match the revision ledger.
- [ ] Exact upstream runtime commit recorded where applicable.
- [ ] Mask ID / special-token assumptions audited.
- [ ] LLaDA, SDAR, and Dream parity artifacts are protocol `trajuq_exact_parity_v3`.
- [ ] Every primary parity artifact covers exactly 20 shared prompts with zero output/provenance mismatches.
- [ ] Every primary parity artifact matches the revision-ledger git commit and model SHA.
- [ ] Pilot output matches the exact state whose trajectory was logged.
- [ ] Precision, CUDA, PyTorch, Transformers, and GPU recorded.

## Engineering
- [ ] 50-item pilot completes on one model.
- [ ] At least 10 trajectory records manually inspected.
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
- [ ] `results/llada_parity.json` reports protocol-v3 PASS for exactly 20 prompts.
- [ ] `results/sdar_parity.json` reports protocol-v3 PASS for exactly 20 prompts.
- [ ] `results/dream_parity.json` reports protocol-v3 PASS for exactly 20 prompts.
- [ ] `results/llada_gsm8k_pilot50_gate.json` reports PASS for exactly 50 raw/scored records.
- [ ] `results/llada_gsm8k_pilot50_trace_review.md` manually reviewed.
- [ ] At least 10 trajectory traces checked for logging/parsing plausibility.
- [ ] `results/math500_scorer_audit.json` reports PASS.
- [ ] `results/math500_scorer_manual_review.json` reports PASS after >=25 calibration-only examples.
- [ ] `results/primary_readiness.json` reports protocol `trajuq_primary_readiness_v2` and PASS.
