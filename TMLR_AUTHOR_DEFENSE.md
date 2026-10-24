# TMLR author defense

This document is a source-grounded rehearsal aid. Answers that depend on unfinished RQ2/RQ4/RQ5 analyses are intentionally bounded rather than guessed.

## Core thesis and novelty

1. **What is the paper's single scientific question?**  What uncertainty remains useful as observability decreases from native diffusion trajectories to an open autoregressive trace and then to black-box repeated behavior?
2. **What is the narrow primary claim?**  A frozen trajectory feature vector adds held-out error-ranking information beyond matched decision-time entropy under the complete 18-cell RQ1 protocol.
3. **Is trajectory-based uncertainty for dLLMs itself new?**  No. The paper cites DiSE and systematic dLLM-UQ work. The contribution is the incremental matched-baseline test, active-unresolved schema, multi-family protocol, and evidence-bound extensions.
4. **Why is decision-time entropy the primary baseline?**  It uses each token's last unresolved distribution immediately before commitment, so it is stronger and better aligned than entropy from an arbitrary final global step.
5. **What does the paper add beyond repeated sampling?**  It evaluates signals from one native generation and explicitly compares their quality/cost with a pre-registered five-sample comparator.
6. **Why five research questions?**  They form one ladder: predictive signal, selective use, compute allocation, architectural transfer, and survival under black-box access.
7. **Could this be framed as several papers?**  Only if the extensions fail to cohere. The manuscript therefore keeps one core question and separates each regime's claims and gates.
8. **What would falsify the main idea?**  Failure to improve the matched decision-time baseline under the registered aggregate and family-direction gate.

## Data and leakage

9. **Where can correctness enter the pipeline?**  Only after immutable generation, through the external scoring stage.
10. **Do raw generation records contain references or correctness labels?**  No; schemas and audits prohibit them.
11. **How are one-split datasets handled?**  By deterministic pre-generation internal holdouts; they are labeled as internal held-out evaluations.
12. **Why cap some datasets?**  To fit the complete registered matrix inside a predeclared compute envelope. Selection uses only namespaced IDs and a fixed SHA-256 ranking seed.
13. **Can caps introduce outcome selection?**  They cannot use labels, references, prompts, or model outcomes under the frozen rule, though they still limit external validity.
14. **Why is MATH-500 not string-scored?**  It uses a pinned symbolic-equivalence scorer and explicit scorer gates.
15. **What happens to ambiguous or failed records?**  They are quarantined or rejected by completeness/audit gates, never silently treated as valid evidence.
16. **Can a completed output be regenerated?**  Repository-supported resume logic reuses valid owned artifacts and generates only missing IDs.

## Model instrumentation and parity

17. **What positions contribute telemetry?**  Only positions active in the current generation block and still unresolved at that step.
18. **Why exclude committed positions?**  Including them would mix post-decision states with uncertainty available before commitment.
19. **How is JS drift aligned?**  It is computed for the same position only when unresolved in both consecutive observations.
20. **What does revision rate measure?**  Top-1 proposal changes while a position remains unresolved, not edits after commitment.
21. **Is the risk score a probability?**  No. The transparent score is a ranking statistic; probability claims use a held-out calibrator.
22. **How do you know instrumentation does not change generation?**  Each adapter must pass exact token-level parity against its pinned uninstrumented reference.
23. **Why are three diffusion families needed?**  They cover masked, mixture-of-experts masked, and blockwise diffusion behavior and expose family heterogeneity.
24. **Can the results be attributed causally to architecture?**  No. Architecture and implementation differ together; cross-family results are predictive and descriptive.

## Statistics and calibration

25. **Why equal-cell macro aggregation?**  It prevents larger benchmarks from dominating the estimand.
26. **Why paired class-stratified bootstrap?**  Methods are evaluated on the same examples, and error/correct classes must remain represented within each cell.
27. **What is the confirmatory endpoint?**  Macro AUROC improvement of trajectory calibration over matched decision-time entropy.
28. **Why require two of three families to improve?**  To block a positive average driven entirely by one family.
29. **Does higher AUROC imply better calibration?**  No. Ranking, probability calibration, and selective risk are reported separately.
30. **Why report ECE sensitivity?**  ECE depends on binning, so 10-, 15-, and 20-bin views prevent a single arbitrary partition from carrying the claim.
31. **How is adaptive-compute thresholding protected from test leakage?**  Thresholds use out-of-fold calibration predictions; the final calibrator and threshold are then applied unchanged to test.
32. **How are multiple black-box comparisons handled?**  The prospective RQ5 plan uses paired inference and Holm correction.
33. **What happens to negative or heterogeneous results?**  They remain in the complete registry and manuscript; they do not trigger feature redesign on test labels.

## Observability regimes

34. **What is observable in open diffusion?**  Stepwise distributions, unresolved masks, revisions, persistence, and commitment timing.
35. **What is observable in the AR control?**  Token distributions and temporal summaries along an irreversible left-to-right trace.
36. **Why are revision/persistence features absent in AR?**  Committed AR tokens are not remasked or revised, so those constructs are structurally inapplicable rather than zero.
37. **What is observable in the black-box regime?**  Response text, repeated-answer behavior, explicit confidence, refusal, length, usage, latency, and optional provider-visible metadata.
38. **Is provider reasoning metadata treated as chain of thought?**  No. It is an interface-visible field with no mechanistic-faithfulness claim.
39. **Can AUROC values be ranked across regimes?**  Only descriptively unless outcome, partition, metric, and aggregation contracts match exactly.
40. **What does the observability ladder establish?**  It tests which registered predictors retain utility under progressively restricted interfaces; it does not isolate architecture as a causal variable.

## Reproducibility and operations

41. **What is the authoritative experiment version?**  Git commit `28002055c828c6910840de6f2105aa6367514144` in the untouched experiment checkout.
42. **How are manuscript numbers produced?**  Generated LaTeX macros are rendered from schema-validated, hash-bound artifacts; values are not copied by hand.
43. **What prevents partial RQ2 progress from entering claims?**  Exact-complete seed/ID validation, provenance and uniqueness audits, statistical gates, and the claim/evidence matrix.
44. **What can reviewers reproduce without gated data or model weights?**  Source audits, synthetic fixtures, artifact validation, analyses over released derived artifacts, tables, figures, and deterministic packaging.
45. **What cannot be fully reproduced?**  Gated rows, provider internals, mutable hosted routing, and potentially bitwise-identical kernels on future hardware.
46. **How is compute reported?**  Result-producing GPU wall time is multiplied by GPU count; utilization is separate. Unknown historical diagnostics are not assigned false precision.
47. **Why not maximize displayed GPU utilization?**  Validated evidence/hour is the objective; unvalidated concurrency can change numerics, duplicate shards, or slow the authoritative writer.
48. **How does scale-out avoid duplicates?**  Only independently qualified A100s receive disjoint untouched shards with immutable worker/shard ownership and exact-complete merge validation.

## Ethics, scope, and disclosure

49. **Does a calibrated score mean probability of truth?**  No. It is task-, scorer-, partition-, and distribution-dependent.
50. **What are the main construct limits?**  Answer correctness does not cover factuality, safety, preference, or open-ended usefulness.
51. **Are benchmark data redistributed?**  No. Gated/raw datasets, caches, credentials, and model weights are excluded.
52. **How was AI used in preparing the work?**  As an assistive tool for editing, code support, and checklists; authors verify and own all ideas, evidence, claims, and text.
53. **What is still scientifically incomplete?**  Final RQ2, RQ4 analysis integration, RQ5, and the cross-regime synthesis remain unauthorized until their registered gates pass.
54. **What should happen if an unfinished result contradicts the narrative?**  The narrative must change; the frozen evidence and negative/null outcome take precedence.

