# TrajUQ experiment protocol

This is the pre-results contract for the ARR/NAACL study. Change it only through an explicit protocol-version commit.

## Unit of analysis

One benchmark item × one model × one decoding configuration × one seed is one generation record.

Each record contains prompt hash/item ID, model/checkpoint/revision, decoding parameters/seed, generated text, compact trajectory summaries, pooled pre-outcome features, runtime metadata, and a correctness label added **after** generation.

## Primary uncertainty families

**Final-state baselines**
- 1 − final mean top-1 confidence
- final normalized entropy
- negative final top-2 margin

**Trajectory features**
- mean / p90 / max normalized entropy
- entropy trend from early to late denoising
- mean / minimum margin
- total top-1 revision rate
- late revision rate
- mask-fraction area under trajectory
- late mask fraction
- normalized mean / p90 commitment time

Features are computed without correctness labels.

## Primary outcomes

Error detection: AUROC, AUPRC.

Calibration: Brier score, NLL, ECE plus bin-count sensitivity.

Selective generation: risk–coverage, AURC, risk at fixed coverage.

Efficiency: denoising steps/NFE, wall-clock latency, throughput, peak GPU memory.

## Calibration

Fit calibrators on a dedicated calibration partition only. Test labels must not select features, calibration family, thresholds, or adaptive-compute rules.

Default: standardized logistic regression. Isotonic is a preregistered nonlinear sensitivity analysis when calibration data are sufficient.

## Adaptive compute

At a base budget, obtain trajectory risk. Intermediate-risk examples receive additional denoising/remasking compute. Compare to fixed schedules at matched mean compute.

Do not call a policy better merely because it spends more compute.

## Pilot gate

Before the primary matrix:
1. run 50 items on one model;
2. manually inspect at least 10 raw records;
3. verify no label leakage;
4. verify deterministic feature recomputation;
5. verify metrics on synthetic edge cases;
6. freeze schema v1;
7. only then scale.

## Falsification criteria

We weaken/reject the core claim if trajectory features do not improve held-out error detection over final confidence, gains vanish under proper calibration, adaptive compute fails at matched compute, or the signal is isolated to one task/model family.
