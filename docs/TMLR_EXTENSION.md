# TMLR extension: cross-paradigm uncertainty dynamics

## Scientific role

This extension strengthens the journal version without rewriting the original
frozen diffusion-LM study. The 3 x 6 LLaDA/SDAR/Dream matrix, its RQ1 claim
gate, RQ2 repeated-sampling comparison, and RQ3 adaptive-compute policy remain
unchanged. RQ4 is a prospective external-validity study registered before its
test outcomes are inspected.

## RQ4

**Which inference-time uncertainty dynamics are shared between diffusion and
autoregressive language models, which are paradigm-specific, and how well do
calibrated risk rankings transfer across architectures?**

The autoregressive control is
`meta-llama/Llama-3.1-8B-Instruct` at immutable revision
`0e9e39f249a16976918f6564b8830bc894c89659`. It is chosen because its scale is
closer to the 7B-class diffusion systems than Llama 3.2 3B. Llama 3.2 3B may be
added later only as a separately labeled scale sensitivity.

## Why this is not a fourth primary family

An AR model has no masked-token revision, remasking, or denoising commitment
trajectory. Folding it into the original diffusion-family macro would therefore
change the estimand. RQ4 instead defines a shared feature subspace (token
entropy, margin, selected-token NLL, temporal slope, and volatility) and keeps
diffusion-only features (revision churn, mask persistence, and commitment time)
as explicit paradigm-specific additions.

## Comparisons

1. Fit and test matched held-out calibrators within each paradigm.
2. Compare trajectory summaries with mean token entropy and sequence NLL.
3. Compare single-run telemetry with five-sample semantic disagreement under
   explicit wall-clock and forward-pass accounting.
4. Transfer the shared-feature calibrator diffusion-to-AR and AR-to-diffusion
   without refitting, and report the degradation relative to within-paradigm
   calibration.
5. Report where diffusion-only revision/commitment features recover errors that
   AR-compatible features miss.

## Outcome isolation and inference

Raw telemetry contains prompts, generated outputs, model/runtime provenance,
and pre-commit uncertainty summaries, but no references or correctness labels.
The original frozen calibration/test IDs are reused. Calibrators are fit on the
calibration partition only. All cell comparisons use paired class-stratified
bootstrap intervals; cross-cell summaries give each model-dataset cell equal
weight. Negative and null results are retained.

## Story of the journal paper

The central claim is not merely that diffusion trajectories contain useful
signals. The journal paper asks what the *shape of uncertainty during
generation* reveals across two different generative factorizations. Diffusion
models expose revision and commitment; AR models expose an irreversible token
history. The shared protocol separates generic temporal uncertainty from the
additional information created by iterative denoising, then tests whether that
information supports abstention and compute allocation.

## Execution order

1. Complete the original authoritative 18-cell primary matrix.
2. Execute original RQ1, then release RQ2 and RQ3 in parallel.
3. Freeze the RQ4 source ledger, Llama revision, tokenizer/chat template,
   prompts, telemetry schema, and analysis plan.
4. Run an outcome-blind Llama parity/telemetry canary.
5. Execute the six Llama cells and the shared-feature transfer analyses.
6. Render the TMLR tables, figures, paper, appendix, and reproducibility bundle.

## Submission constraints

The TMLR submission must use the official TMLR style, remain double blind, and
keep the supplement anonymized. The original ACL/ARR package is historical
input only; it is not the submission target for this branch.
