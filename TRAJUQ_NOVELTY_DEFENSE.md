# TrajUQ novelty defense

Status: literature audit current through 2026-10-29. This document is a positioning artifact, not an authorization to alter or report any evidence-locked RQ4/RQ5 result.

## Defensible thesis

TrajUQ should be presented as an **observability-stratified empirical study of uncertainty and its actionability**, not as the invention of trajectory uncertainty or black-box uncertainty:

> TrajUQ studies uncertainty under progressively restricted model-access regimes—from native diffusion-generation trajectories, through autoregressive token traces, to output-only black-box behavioral signals—and evaluates those signals under matched risk-ranking, selective-prediction, calibration, compute-allocation, and cost-aware protocols.

The matrix in `CLOSEST_WORK_MATRIX.tsv` contains 21 verified primary works. It makes three boundaries clear.

1. Trajectory-derived uncertainty is already established in diffusion generation, autoregressive reasoning, and interactive agent trajectories.
2. Confidence evolution is already used for answer selection, decoding acceleration, and online intervention.
3. Sampling-based semantic and consistency uncertainty is already established for black-box or partially black-box models.

The remaining contribution is the **combination and controlled comparison**: what uncertainty remains useful under three explicitly separated access regimes; whether process information adds held-out value beyond a matched endpoint score; whether that value supports abstention and compute allocation; which summaries transfer across generation paradigms; and what additional inference cost is required. This is a scientific-understanding contribution, not a universal priority claim or a claim that rigor itself is algorithmic novelty.

## Independently defensible components

### 1. Observability-stratified evaluation

The study separates rather than conflates:

- native diffusion trajectories: stepwise distributions, remasking, revisions, persistence, and commitment;
- autoregressive traces: irreversible token-by-token confidence evolution without remasking or delayed commitment;
- output-only behavior: repeated responses, semantic/consistency signals, latency, usage, and only metadata actually exposed by the provider.

The defensible point is not that these uncertainty families are new. It is that they are evaluated as distinct access regimes within one evidence-gated program. Cross-regime summaries must remain descriptive unless the frozen matched-comparison policy authorizes a statistical comparison.

### 2. Incremental information beyond an endpoint

The central falsifiable test is nested: identical examples, splits, calibration procedure, and endpoint feature; the candidate adds only frozen trajectory features. This supports a bounded statement about **incremental held-out error-ranking information**. It does not support a statement that trajectories are universally superior, causal, or intrinsically better calibrated.

### 3. Selective utility

AUROC alone is not enough. TrajUQ connects uncertainty to abstention and risk–coverage behavior under held-out evaluation. Selective generation and semantic-UQ papers already establish the general value of abstention; TrajUQ's contribution is to ask which access-specific signal supports it under matched within-regime protocols.

### 4. Compute allocation

Trajectory-triggered sparse collaborative decoding, DiSE, OSDT, CD4LM, STDec, and PACE-dLLM already show that confidence dynamics can control inference. TrajUQ must therefore claim only its frozen policy question: whether calibrated answer-level risk allocates a registered second pass more effectively than matched random allocation under the study's budget and evidence gate.

### 5. Cross-paradigm transfer

Diffusion remasking, persistence, revisions, and delayed commitment have no literal AR analogue. Transfer claims must concern explicitly mapped summaries, calibrated rankings, or model reuse specified by the frozen protocol—not an assertion that an AR trace is a denoising trajectory. A negative or heterogeneous transfer result is informative and must be retained.

### 6. Cost–observability trade-off

The synthesis has two axes:

- **access:** native trajectory → token trace → output-only behavior;
- **additional inference budget:** one generation → selective second pass → repeated samples.

The paper may relate performance to measured wall time, number of generations, target tokens, or ledger-bound API cost where those quantities are authoritative. It must not infer costs from list prices, mix incompatible protocols, or claim a causal monotonic law of observability.

### 7. Evidence discipline

Frozen partitions, calibration-only fitting, registered hypotheses, paired confidence intervals, equal-cell aggregation, multiplicity control, provenance binding, and retained negative results strengthen the credibility of the study. They are not themselves a new UQ algorithm.

## Claims TrajUQ must not make

- “TrajUQ is the first method to use uncertainty trajectories.”
- “TrajUQ introduces trajectory-level uncertainty.”
- “TrajUQ is the first to use confidence evolution to predict reasoning correctness.”
- “TrajUQ is the first to use trajectory uncertainty for adaptive inference or intervention.”
- “TrajUQ introduces black-box uncertainty estimation.”
- “TrajUQ is the first semantic- or consistency-based black-box UQ method.”
- “Reducing observability reduces uncertainty quality.” The experiments do not intervene on observability while holding models, tasks, and protocols fixed.
- “Trajectory features improve calibration” unless the authorized calibration endpoints pass their own gate. Better ranking or AURC does not imply better ECE/NLL.
- “Cross-paradigm transfer is universal.” Transfer may be feature-, family-, and dataset-dependent.
- “Near-full-refinement performance is equivalent to full refinement” without a registered equivalence/non-inferiority test.
- Any RQ4/RQ5 scientific statement derived from partial, unreconciled, or unauthorized artifacts.

## Closest-work stress test

### Direct trajectory and intervention threats

- [STEP](https://arxiv.org/abs/2610.34537) scores each AR reasoning step and extracts soft confidence-trajectory features for correctness prediction. It defeats any broad priority claim around stepwise reasoning confidence or temporal correctness estimation.
- [Trajectory-Triggered Sparse Collaborative Decoding](https://arxiv.org/abs/2610.23613) uses partial confidence-trajectory shape to predict failure and trigger bounded target-model intervention. It defeats priority claims around early trajectory actionability and risk-guided extra compute.
- [Inference Time Optimization with Confidence Dynamics](https://arxiv.org/abs/2605.25244) uses confidence gain along reasoning traces for inference-time voting. It defeats claims that confidence evolution first improves answer selection.
- [Uncertainty Quantification for Large Language Diffusion Models](https://arxiv.org/abs/2605.14570) studies intermediate generations, remasking, denoising complexity, masked likelihood, semantic trajectory dissimilarity, selective error detection, and cost. It is the closest diffusion-UQ threat and requires TrajUQ to emphasize its nested endpoint-incrementality test and cross-regime/actionability program.
- [DiSE](https://aclanthology.org/2026.acl-long.298/) uses sequence-regeneration probability for dLLM self-evaluation and adaptive length. It defeats claims around first dLLM confidence or first uncertainty-guided diffusion compute.
- [OSDT](https://arxiv.org/abs/2511.02077), [CD4LM](https://arxiv.org/abs/2601.02236), [STDec](https://arxiv.org/abs/2604.06330), and [PACE-dLLM](https://arxiv.org/abs/2609.26249) all exploit stepwise confidence or temporal stability for efficient diffusion decoding. Their objectives differ from answer-level held-out error prediction, but they foreclose novelty claims about confidence-aware dLLM acceleration.
- [Beyond Single-Turn Confidence](https://arxiv.org/abs/2608.11552) compares token-probability, resampled-trajectory, and reflexive UQ for tool-using agent trajectories. It means “multiple observability families at trajectory level” is also too broad as a priority claim; TrajUQ's scope must remain native language generation across diffusion, AR, and hosted output-only regimes.

### Black-box and semantic threats

- [BiG-SURE](https://arxiv.org/abs/2608.30646) is an explicitly black-box, cross-temperature semantic-agreement estimator evaluated with abstention AUROC.
- [SelfCheckGPT](https://aclanthology.org/2023.emnlp-main.557/) establishes zero-resource sampling-based black-box consistency for hallucination/factuality ranking.
- [Semantic Uncertainty](https://arxiv.org/abs/2302.09664), [Semantic Entropy](https://doi.org/10.1038/s41586-024-07421-0), [Kernel Language Entropy](https://doi.org/10.52202/079017-0283), and [Semantic Density](https://doi.org/10.52202/079017-4274) establish strong meaning-aware repeated-sampling baselines.
- [MARS](https://doi.org/10.1109/ISIT57864.2024.10619136) establishes meaning-aware token-likelihood aggregation; [ConU](https://arxiv.org/abs/2407.00499) adds conformal correctness/coverage control; [Selective Generation](https://doi.org/10.52202/079017-1600) establishes textual-entailment-based risk control.

### Calibration and interpretation threats

- [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221) is a core correctness-prediction/self-evaluation precedent.
- [The Mirage of Calibrated Confidence](https://arxiv.org/abs/2609.18453) warns that ordinary calibration and AUROC can fail to measure whether confidence is actually grounded in the trajectory. TrajUQ should treat nested trajectory-versus-endpoint evaluation as a bounded predictive test, not proof of mechanistic grounding.

## Verification disposition for requested names

- **STEP:** verified on arXiv, submitted 2026-10-25; included.
- **Trajectory-Triggered Sparse Collaborative Decoding:** verified on arXiv, submitted 2026-10-16; included.
- **BiG-SURE:** verified on arXiv with EMNLP 2026 main-conference journal reference; included.
- **SCOPE: Stepwise Calibration of Reasoning Confidence with Outcome-Grounded Prefix Evidence:** **not verified**. Exact-title and subtitle searches performed on 2026-10-29 did not locate an arXiv, DOI, ACL Anthology, OpenReview, or publisher record. It is excluded from the evidence-bearing matrix until a primary URL, DOI, or manuscript identifier is supplied. Do not cite or characterize it from the directive alone.

## Recommended manuscript language

### Abstract/Introduction

> We study what uncertainty remains useful as access to language generation changes: from native diffusion trajectories, through autoregressive token traces, to output-only black-box behavior. Rather than proposing trajectory or semantic uncertainty as new categories, we test their incremental predictive value, selective utility, compute-allocation value, transfer, and measured inference cost under access-matched protocols.

### Contribution list

1. An observability-stratified empirical framework separating native diffusion trajectories, AR token traces, and output-only behavioral signals.
2. A nested held-out test of whether process features add information beyond a calibrated endpoint baseline on identical examples.
3. Matched evaluation of error ranking, risk–coverage behavior, and registered adaptive-compute policies, with inference cost reported only from authoritative measurements.
4. A transfer analysis that distinguishes shared temporal summaries from diffusion-specific revision, persistence, and commitment signals.
5. Evidence-gated synthesis with explicit heterogeneity and negative results; this is methodological discipline, not a standalone novelty claim.

### Discussion/Limitations

> The access regimes differ in models, interfaces, and available signals; their comparison is descriptive except where the frozen matched-comparison policy applies. The study therefore does not identify a causal effect of reduced observability. Its conclusions are bounded to the evaluated families, tasks, budgets, and calibration distributions.

## Audit rule

Every literature claim in the matrix is tied to a primary paper or venue page. `Yes`, `No`, and `Partial` record what the primary source supports; `Partial` is used when only some method variants satisfy the column. Absence means “not reported as a contribution in the verified primary source,” not proof that no appendix experiment exists. The matrix should be rechecked immediately before submission because several closest works are October 2026 preprints.
