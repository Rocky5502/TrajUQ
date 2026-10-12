# Frozen analysis plan (pre-results)

This file is a pre-results contract. Changes that affect primary claims must be committed before primary test outcomes are inspected.

## Active-position semantics

All entropy, confidence, and margin statistics are computed only on positions that are both **active for the current denoising block** and **still unresolved at the pre-transfer observation**. Logits on prompt tokens, future inactive blocks, or already committed tokens are excluded. Revision rate counts prediction changes only for positions unresolved in two consecutive observations. Jensen-Shannon drift is computed exactly from consecutive full-vocabulary predictive distributions on those same eligible positions and normalized by \(\log 2\), so it lies in \([0,1]\). Only one previous probability tensor is retained, bounding logging memory to one extra generated-length-by-vocabulary distribution. Commitment time is normalized within each position's active window. These schema-v4 semantics are frozen before primary outcomes are inspected.

## Blockwise SDAR trajectories

For SDAR, each active block emits mergeable position- and step-level sufficient statistics. Global percentiles, extrema, late-stage rates, temporal slopes, and decision-time features are recomputed after concatenating blocks in generation order; block-level percentiles/slopes are never averaged. This keeps the primary feature semantics aligned across whole-sequence and blockwise diffusion.

## Primary endpoint and claim gate

The single primary RQ1 endpoint is the equal-cell-weight macro improvement in AUROC of the trajectory-augmented logistic model over the decision-time entropy logistic baseline. AUPRC and AURC improvements are secondary endpoints. A positive RQ1 claim requires the 95\% paired class-stratified within-cell bootstrap interval for macro AUROC improvement to lie above zero and positive point improvement in at least two of the three primary model families. Cell-level and secondary results are reported regardless of direction; this gate is frozen to prevent choosing the most favorable metric after test inspection. The primary macro aggregator rejects an incomplete model--dataset cell set unless `--allow-partial` is explicitly used for exploration, and `scripts/evaluate_rq1_claim_gate.py` converts the frozen rules into a machine-readable claim decision.

## RQ1 — Does the trajectory add predictive information?

The primary incremental-information test compares two calibration-fitted predictors on identical held-out examples: (i) a per-token **decision-time entropy** baseline, defined as mean normalized entropy at the last unresolved observation immediately before each token is committed, and (ii) a nested logistic model containing that exact baseline feature plus the frozen trajectory feature vector. This directly tests whether the trajectory adds information beyond a strong final-decision confidence baseline and avoids the unstable alternative of using only the last global denoising step, which may contain very few unresolved tokens. Primary discrimination metrics are AUROC and AUPRC; the primary selective metric is discrete AURC. Use paired, class-stratified bootstrap confidence intervals. All reported improvement deltas are oriented so positive means the trajectory-augmented method is better.

The transparent training-free TrajUQ raw-risk score is retained as a diagnostic/sensitivity baseline, not the primary RQ1 test.

A claim that the trajectory adds information requires a meaningful held-out improvement that is not confined to a single dataset/model family.

## Frozen feature-family ablations

Before primary outcomes are inspected, the following ablations are frozen: decision-only; decision plus distribution-trajectory features; decision plus stability features (JS drift and revisions); decision plus persistence/commitment features; the full feature set; and leave-family-out variants removing JS, revisions, or persistence. Every variant is fit only on the calibration partition and evaluated on the identical held-out test examples. Paired bootstrap deltas are always reported relative to the decision-only baseline.

## Expensive repeated-sampling baseline

As a stronger but more expensive uncertainty comparator, freeze a five-sample final-answer disagreement baseline using seeds 2027--2031, native sampling temperature 0.2, and top-$p=1.0$. For each example, independently generated label-free samples are canonicalized without consulting the reference answer; variation ratio, normalized vote entropy, and unique-answer fraction are calibrated on the calibration partition and evaluated on the same held-out target labels. Its extra inference cost must be reported explicitly. The evaluator requires the same five seed IDs and identical generation configuration on calibration and test, verifies that sample artifacts contain no outcome labels, and reports auxiliary repeated-generation wall time per target example. The final 18-cell repeated-sampling artifact requires complete test-time wall-time metadata in every cell and reports the equal-cell-weight mean auxiliary seconds per example (target generation excluded), so compute accounting cannot be silently dropped during macro aggregation. Paired held-out comparisons against \method use the exact same test IDs and labels. For MATH-500, the five sampled answers are clustered without reference answers using the same pinned Math-Verify equivalence relation as the correctness scorer. All ten sample pairs are checked, connected components define vote classes, and exact text equality is accepted before symbolic parsing. This enables the repeated-sampling comparator on the full frozen 18-cell matrix without unsafe string matching.

## RQ2 — Calibration and selective generation

Fit probability calibrators on the frozen calibration partition only. Primary calibrator: standardized logistic regression over the frozen feature vector. Isotonic calibration is supplementary only: it post-calibrates the same standardized logistic classifier using five-fold stratified out-of-fold scores with `ensemble=False`, never test labels. A model--dataset cell is eligible only when its calibration partition contains at least 1000 examples and at least five examples in each correctness class; otherwise the frozen analysis emits `SKIP_INSUFFICIENT_CALIBRATION`. The complete sensitivity macro retains all 18 cells as either validated PASS or validated SKIP and macro-averages isotonic-vs-logistic calibration deltas only over the pre-registered eligible subset. This sensitivity cannot alter the primary logistic RQ1 endpoint or claim gate.

Report:
- Brier score;
- NLL;
- ECE at 15 bins plus 10-bin and 20-bin sensitivity;
- risk–coverage and AURC;
- risk at frozen coverage points.

For cross-cell RQ2 summaries, report equal-cell-weight macro differences for Brier, NLL, 15-bin ECE, AURC, and risk at 50\% and 80\% coverage using the same paired class-stratified within-cell bootstrap, with positive improvement always defined as better (therefore baseline minus TrajUQ for lower-is-better metrics).

Raw TrajUQ risk must not be described as an error probability.

## RQ3 — Adaptive compute

Freeze two complete-generation budgets per model: LLaDA and Dream use 32 versus 128 denoising steps, while SDAR uses 2 versus 4 denoising steps per block. The primary policy cap is 50\% selected examples; 25\% and 75\% are sensitivity settings. Compute risk from the completed base-budget answer. On calibration data, select a refinement threshold under the frozen cap and apply it unchanged to test.

Compare against:
1. fixed base compute;
2. fixed refined compute;
3. random refinement at the same expected fraction.

Model-native confidence/remasking schedulers are not part of the frozen primary
RQ3 comparator because they are not available under one matched policy contract
across all three primary model families. Any future model-specific scheduler
analysis must be labeled exploratory and cannot replace the frozen comparisons
above.

The primary execution semantics are a **selective second-pass rerun**: every example pays the complete base generation, and a selected high-risk example launches a separate full refined-budget generation from scratch. Thus mean nominal compute is $B_0 + fB_1$, not $B_0+f(B_1-B_0)$, where $f$ is the selected fraction. This is directly realizable from paired frozen artifacts and does not pretend that an independently generated refined answer is a continuation of the base trajectory. A true continuation policy is reserved for future model-specific runtime support.

For the frozen TrajUQ policy, report paired bootstrap confidence intervals for accuracy change versus the base budget, versus fixed refined generation, and versus the expected random second-pass policy at the same selected fraction. The threshold remains fixed from calibration during all test resamples. Base/refined artifacts must match in dataset, prompt hash, model, adapter, seed, revision, scorer, and all generation settings except the compute budget. In addition to nominal denoising budget, report measured wall time and model-forward evaluations under the same second-pass accounting whenever those metadata are complete.

The primary RQ3 summary macro-averages all 18 model--dataset cells with equal cell weight. Its confidence intervals use a paired within-cell bootstrap followed by macro-averaging on each replicate. Report adaptive accuracy, deltas versus base/refined/random, selected fraction, and nominal cost ratio to the base budget. The 50\% cap is primary; the 25\% and 75\% caps are reported as frozen sensitivity analyses rather than alternative thresholds chosen after test inspection.

The repository's `evaluate_adaptive.py` implements the per-cell policy and exports per-example decisions; `aggregate_adaptive.py` implements the full-matrix equal-cell macro summary with paired within-cell resampling. Model-native schedulers remain model-specific.

## RQ4 — Cross-paradigm uncertainty dynamics (prospective TMLR extension)

RQ4 asks which inference-time uncertainty dynamics are shared between diffusion
and autoregressive language models, which are paradigm-specific, and how well
calibrated risk rankings transfer across architectures. The AR control is
Llama-3.1-8B-Instruct at the immutable revision recorded in
`configs/models/llama31_ar.yaml`, evaluated on the same frozen calibration and
test IDs as the original six-dataset matrix.

The shared feature space contains mean/max/late token entropy, mean selected-token
negative log likelihood, minimum margin, entropy slope, and entropy volatility.
Diffusion-only revision churn, JS drift, mask persistence, and commitment timing
remain separate additions and are never imputed for Llama. Report within-paradigm
held-out calibration, single-run versus five-sample semantic disagreement, and
unrefit diffusion-to-AR and AR-to-diffusion transfer. The primary RQ4 endpoint is
equal-cell macro AURC improvement over mean token entropy; AUROC, AUPRC, Brier,
NLL, ECE, and fixed-coverage risks are secondary.

RQ4 is prospectively registered and cannot alter the original RQ1 claim gate.
AR output is not ground truth, test outcomes cannot select features, and null or
negative transfer results must be reported. The full contract is in
`configs/experiments/tmlr_rq4.yaml` and `docs/TMLR_EXTENSION.md`.

## Selective-risk convention

Discrete AURC is the mean selective risk over the attainable empirical coverage levels \(k/n\), \(k=1,\ldots,n\). Lower is better. Exact score ties use the expected cumulative error under a uniform random ordering within each tie group, so AURC and fixed-coverage risk cannot depend on benchmark row order. Fixed-coverage risk uses the smallest attainable coverage at or above the target. This tie-aware convention is frozen before primary outcomes are inspected.

## Confidence intervals and aggregation

Use paired class-stratified bootstrap resampling over examples. For cross-model/cross-dataset summaries, compute each metric within every model--dataset cell and give cells equal macro weight. For confidence intervals, resample examples class-stratified within each cell on every bootstrap replicate and then macro-average the cell-level improvements. Do not pool heterogeneous tasks merely to obtain tighter confidence intervals. Report how many cells have positive, zero, and negative held-out improvement.

## Multiple seeds

Sampling-sensitive configurations require repeated-seed sensitivity analysis. Do not choose a favorable seed after test inspection.

## Negative results

Do not change the frozen feature dictionary, dataset subset, calibrator family, or adaptive threshold rule because the primary test result is weak. If trajectory information does not improve on final confidence, report the negative result and narrow the claim.
