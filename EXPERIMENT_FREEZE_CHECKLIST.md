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
- [ ] Exact model and tokenizer revision recorded.
- [ ] Exact upstream runtime commit recorded where applicable.
- [ ] Mask ID / special-token assumptions audited.
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


## Pilot automation artifacts
- [ ] `results/llada_parity.json` reports PASS.
- [ ] `results/llada_gsm8k_pilot50_gate.json` reports PASS.
- [ ] `results/llada_gsm8k_pilot50_trace_review.md` manually reviewed.
- [ ] At least 10 trajectory traces checked for logging/parsing plausibility.
