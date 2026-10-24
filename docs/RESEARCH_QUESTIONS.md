# Research questions and evidence boundaries

TrajUQ separates the original diffusion study from later external-validity extensions. This document describes the scientific question each stage answers; it does not turn execution progress into a claim.

## RQ1 — Incremental predictive signal

**Question.** Do native denoising-trajectory signals predict eventual answer error beyond the same model's calibrated decision-time uncertainty?

**Primary comparison.** Standardized logistic calibration using decision-time entropy versus the same calibrated model augmented with the frozen trajectory feature vector.

**Primary endpoint.** Equal-cell macro AUROC improvement across the frozen 3-model × 6-dataset matrix.

**Positive-claim gate.** The paired confidence interval lower bound must be greater than zero and the point improvement must be positive in at least two of the three primary diffusion families.

**Boundary.** A positive aggregate result does not imply uniform gains for every model, dataset, calibration metric, or mechanism.

## RQ2 — Selective generation

**Question.** Does calibrated trajectory risk improve held-out error ranking and risk–coverage behavior over the registered comparators?

**Comparator.** A five-sample repeated-generation baseline on the pre-registered 3-model × 4-dataset subset: GSM8K, MATH-500, GPQA-Diamond, and MMLU-Pro.

**Repeated-sampling seeds.** 2027–2031.

**Sampling contract.** Temperature 0.2 and top-p 1.0 for the diffusion repeated-sampling baseline.

**Boundary.** ARC-Challenge and TruthfulQA remain part of the full primary study but are excluded from this compute-bounded repeated-sampling comparator by the frozen protocol.

## RQ3 — Adaptive compute

**Question.** Can calibrated trajectory risk allocate a second generation budget more effectively than fixed or matched-random refinement?

**Policy selection.** Thresholds are selected from out-of-fold calibration predictions only.

**Primary cap.** Refine at most 50% of examples; 25% and 75% are registered sensitivity caps.

**Budgets.**
- LLaDA: 32 → 128 steps
- Dream: 32 → 128 steps
- SDAR: 2 → 4 denoising steps

**Boundary.** This evaluates registered second-pass policies, not arbitrary dynamic scheduling.

## RQ4 — Autoregressive control

**Question.** Which inference-time uncertainty dynamics are shared between diffusion and autoregressive language models, which are architecture-specific, and how well do calibrated risk rankings transfer across architectures?

**Model.** Meta Llama 3.1 8B Instruct, prospectively frozen as a separate autoregressive control.

**Important separation.** Llama 3.1 is not retroactively inserted into the original RQ1 diffusion-family gate.

**Cross-regime interpretation.** Descriptive unless outcome definitions, partitions, metrics, and aggregation rules are matched.

## RQ5 — Black-box behavioral uncertainty

**Question.** How much useful uncertainty remains when models are accessible only through a hosted API and internal probabilities are unavailable?

**Models.** DeepSeek-V3.2, GLM-5.1, and Kimi-K2.6 through their frozen provider-visible identifiers.

**Signals.** Repeated-sampling disagreement, self-consistency, semantic response consistency, verbalized confidence, answer stability, abstention/refusal, and provider-visible diagnostics.

**Boundary.** Provider-visible reasoning metadata are not hidden activations and are not treated as faithful chain-of-thought evidence.

## Evidence-state rule

For every RQ:

~~~text
planned → generated → saved → audited → scored → statistically complete
→ gate evaluated → claim authorized → reported
~~~

A later state must never be inferred from an earlier one. Authoritative artifacts and frozen registries take precedence over progress messages or stale documentation.
