# Experiment freeze checklist

## Scientific definition
- [ ] RQ1–RQ4 wording frozen.
- [ ] Primary model families frozen.
- [ ] Primary datasets/splits frozen.
- [ ] Primary trajectory features/baselines frozen.
- [ ] Calibration split/method frozen.
- [ ] Primary metrics/statistics frozen.
- [ ] Adaptive policy and matched-compute comparison frozen.

## Engineering
- [ ] 50-item pilot completes on one model.
- [ ] At least 10 trajectory records manually inspected.
- [ ] No NaN/Inf in features.
- [ ] Duplicate IDs rejected.
- [ ] Resume logic verified.
- [ ] Prompt/reference separation audited.
- [ ] Seed/model/dataset revisions recorded.
- [ ] Timing/GPU memory recorded.
- [ ] Output SHA-256 recorded.
- [ ] Unit tests pass at experiment commit.

## Integrity
- [ ] Correctness labels added only after generation.
- [ ] Test labels not used to select features/thresholds/calibrator.
- [ ] No gated benchmark rows committed.
- [ ] No secrets/tokens committed.
- [ ] No paper number without a traceable artifact.
