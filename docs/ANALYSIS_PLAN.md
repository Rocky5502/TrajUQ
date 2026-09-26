# Frozen analysis plan (pre-results)

This file is a pre-results contract. Changes that affect primary claims must be committed before primary test outcomes are inspected.

## RQ1 — Does the trajectory add predictive information?

The primary incremental-information test compares two calibration-fitted predictors on identical held-out examples: (i) a final-state entropy baseline and (ii) a nested logistic model containing the same final entropy plus the frozen trajectory feature vector. This directly tests whether the trajectory adds information beyond the final state. Primary discrimination metrics are AUROC and AUPRC; the primary selective metric is discrete AURC. Use paired, class-stratified bootstrap confidence intervals. All reported improvement deltas are oriented so positive means the trajectory-augmented method is better.

The transparent training-free TrajUQ raw-risk score is retained as a diagnostic/sensitivity baseline, not the primary RQ1 test.

A claim that the trajectory adds information requires a meaningful held-out improvement that is not confined to a single dataset/model family.

## RQ2 — Calibration and selective generation

Fit probability calibrators on the frozen calibration partition only. Primary calibrator: standardized logistic regression over the frozen feature vector. Isotonic regression is a nonlinear sensitivity analysis when calibration data are sufficient.

Report:
- Brier score;
- NLL;
- ECE at 15 bins plus 10-bin and 20-bin sensitivity;
- risk–coverage and AURC;
- risk at frozen coverage points.

Raw TrajUQ risk must not be described as an error probability.

## RQ3 — Adaptive compute

Freeze at least two compute budgets per model. Compute risk at the base budget. On calibration data, select a refinement threshold under a preregistered maximum refinement fraction. Apply the threshold unchanged to test.

Compare against:
1. fixed base compute;
2. fixed refined compute;
3. random refinement at the same expected fraction;
4. native confidence/remasking scheduling where the model exposes it.

The repository's `evaluate_adaptive.py` implements the two-budget TrajUQ-versus-random comparison. Model-native schedulers remain model-specific.

## RQ4 — Sampler/denoiser diagnostics

Vary denoising budgets/remasking settings and measure calibration transfer. For FLARE/HybridDiffusion, if both paths are instrumented, treat AR/diffusion disagreement as a diagnostic feature only. AR output is not ground truth.

## Selective-risk convention

Discrete AURC is the mean selective risk over the attainable empirical coverage levels \(k/n\), \(k=1,\ldots,n\). Lower is better. Fixed-coverage risk uses the smallest attainable coverage at or above the target. This definition is frozen before primary outcomes are inspected.

## Confidence intervals and aggregation

Use paired class-stratified bootstrap resampling over examples. For cross-dataset summaries, compute metrics within each dataset first and macro-average; do not pool heterogeneous tasks merely to obtain tighter confidence intervals.

## Multiple seeds

Sampling-sensitive configurations require repeated-seed sensitivity analysis. Do not choose a favorable seed after test inspection.

## Negative results

Do not change the frozen feature dictionary, dataset subset, calibrator family, or adaptive threshold rule because the primary test result is weak. If trajectory information does not improve on final confidence, report the negative result and narrow the claim.
