# Frozen analysis plan (pre-results)

This file is a pre-results contract. Changes that affect primary claims must be committed before primary test outcomes are inspected.

## Active-position semantics

All entropy, confidence, and margin statistics are computed only on positions that are both **active for the current denoising block** and **still unresolved at the pre-transfer observation**. Logits on prompt tokens, future inactive blocks, or already committed tokens are excluded. Revision rate counts prediction changes only for positions unresolved in two consecutive observations. Jensen-Shannon drift is computed exactly from consecutive full-vocabulary predictive distributions on those same eligible positions and normalized by \(\log 2\), so it lies in \([0,1]\). Only one previous probability tensor is retained, bounding logging memory to one extra generated-length-by-vocabulary distribution. Commitment time is normalized within each position's active window. These schema-v4 semantics are frozen before primary outcomes are inspected.

## Blockwise SDAR trajectories

For SDAR, each active block emits mergeable position- and step-level sufficient statistics. Global percentiles, extrema, late-stage rates, temporal slopes, and decision-time features are recomputed after concatenating blocks in generation order; block-level percentiles/slopes are never averaged. This keeps the primary feature semantics aligned across whole-sequence and blockwise diffusion.

## Primary endpoint and claim gate

The single primary RQ1 endpoint is the equal-cell-weight hierarchical macro improvement in AUROC of the trajectory-augmented logistic model over the decision-time entropy logistic baseline. AUPRC and AURC improvements are secondary endpoints. A positive RQ1 claim requires the 95\% hierarchical bootstrap interval for macro AUROC improvement to lie above zero and positive point improvement in at least two of the three primary model families. Cell-level and secondary results are reported regardless of direction; this gate is frozen to prevent choosing the most favorable metric after test inspection.

## RQ1 — Does the trajectory add predictive information?

The primary incremental-information test compares two calibration-fitted predictors on identical held-out examples: (i) a per-token **decision-time entropy** baseline, defined as mean normalized entropy at the last unresolved observation immediately before each token is committed, and (ii) a nested logistic model containing that exact baseline feature plus the frozen trajectory feature vector. This directly tests whether the trajectory adds information beyond a strong final-decision confidence baseline and avoids the unstable alternative of using only the last global denoising step, which may contain very few unresolved tokens. Primary discrimination metrics are AUROC and AUPRC; the primary selective metric is discrete AURC. Use paired, class-stratified bootstrap confidence intervals. All reported improvement deltas are oriented so positive means the trajectory-augmented method is better.

The transparent training-free TrajUQ raw-risk score is retained as a diagnostic/sensitivity baseline, not the primary RQ1 test.

A claim that the trajectory adds information requires a meaningful held-out improvement that is not confined to a single dataset/model family.

## Frozen feature-family ablations

Before primary outcomes are inspected, the following ablations are frozen: decision-only; decision plus distribution-trajectory features; decision plus stability features (JS drift and revisions); decision plus persistence/commitment features; the full feature set; and leave-family-out variants removing JS, revisions, or persistence. Every variant is fit only on the calibration partition and evaluated on the identical held-out test examples. Paired bootstrap deltas are always reported relative to the decision-only baseline.

## Expensive repeated-sampling baseline

As a stronger but more expensive uncertainty comparator, freeze a five-sample final-answer disagreement baseline using seeds 2027--2031, native sampling temperature 0.2, and top-$p=1.0$. For each example, independently generated label-free samples are canonicalized without consulting the reference answer; variation ratio, normalized vote entropy, and unique-answer fraction are calibrated on the calibration partition and evaluated on the same held-out target labels. Its extra inference cost must be reported explicitly. The evaluator requires the same five seed IDs and identical generation configuration on calibration and test, verifies that sample artifacts contain no outcome labels, and reports auxiliary repeated-generation wall time per target example. Paired held-out comparisons against \method use the exact same test IDs and labels. MATH-500 is excluded from this baseline until algebraic-equivalence clustering is implemented; string matching is not permitted as a substitute.

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

For the frozen TrajUQ policy, report paired bootstrap confidence intervals for accuracy change versus the base budget, versus fixed refined compute, and versus the expected random-refinement policy at the same selected refinement fraction. The threshold remains fixed from calibration during all test resamples.

The repository's `evaluate_adaptive.py` implements the two-budget TrajUQ-versus-random comparison. Model-native schedulers remain model-specific.

## RQ4 — Sampler/denoiser diagnostics

Vary denoising budgets/remasking settings and measure calibration transfer. For FLARE/HybridDiffusion, if both paths are instrumented, treat AR/diffusion disagreement as a diagnostic feature only. AR output is not ground truth.

## Selective-risk convention

Discrete AURC is the mean selective risk over the attainable empirical coverage levels \(k/n\), \(k=1,\ldots,n\). Lower is better. Fixed-coverage risk uses the smallest attainable coverage at or above the target. This definition is frozen before primary outcomes are inspected.

## Confidence intervals and aggregation

Use paired class-stratified bootstrap resampling over examples. For cross-model/cross-dataset summaries, compute each metric within every model--dataset cell and give cells equal macro weight. For confidence intervals, resample examples class-stratified within each cell on every bootstrap replicate and then macro-average the cell-level improvements. Do not pool heterogeneous tasks merely to obtain tighter confidence intervals. Report how many cells have positive, zero, and negative held-out improvement.

## Multiple seeds

Sampling-sensitive configurations require repeated-seed sensitivity analysis. Do not choose a favorable seed after test inspection.

## Negative results

Do not change the frozen feature dictionary, dataset subset, calibrator family, or adaptive threshold rule because the primary test result is weak. If trajectory information does not improve on final confidence, report the negative result and narrow the claim.
