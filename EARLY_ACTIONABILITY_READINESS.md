# Early-actionability readiness

Status: **FEASIBLE FROM EXISTING TELEMETRY; ANALYSIS LOCKED UNTIL FINAL RQ4 AUTHORIZATION**

Classification: `SECONDARY_EXPLORATORY`

This analysis was proposed on 2026-10-29 after partial RQ4 generation existed. It was defined without inspecting partial RQ4 scientific outcomes and is not part of the original H4a/H4b/H4c family. It cannot alter the registered RQ4 claim gate, trigger replacement generation, or be described as prospectively registered before RQ4.

## Telemetry feasibility audit

The audit inspected schemas and generation code, not partial scientific values.

- **Autoregressive regime:** feasible. The frozen Llama adapter stores an ordered pre-token-commit `token_trace` with entropy, margin, selected-token log probability, and top-k token log probabilities for every generated token. Ordered `steps` summaries are also stored.
- **Diffusion regime:** feasible with a deliberately restricted feature space. Raw artifacts store ordered `steps` summaries containing step entropy, entropy p90, confidence, margin, mask fraction, revision fraction, JS drift, active fraction, and active count. These summaries support fixed prefix aggregation.
- **Diffusion limitation:** full per-position vocabulary distributions and the accumulator's mergeable sufficient-statistic arrays are not persisted. Prefix analysis must therefore use only the stored ordered step summaries. It must not claim to reconstruct the complete primary feature vector, exact prefix commitment-time statistics, or per-position prefix percentiles.
- **No new generation:** required telemetry is already present. No model call, replay, replacement trajectory, or telemetry change is authorized.

## Frozen contract

The executable contract is `configs/analysis/rq4_early_actionability_secondary.json`.
Its frozen SHA-256 is `814965b4af71b9e042098d161e5691b0722eccb07c831668932c0d87c00b7194`.

- Fixed fractions: 25%, 50%, 75%, and 100%.
- Prefix length: `max(1, ceil(fraction * stored_trace_length))`.
- Diffusion prefix unit: ordered stored step summary.
- AR prefix unit: ordered generated-token trace.
- Feature definitions are fixed in the contract and use only observations within the selected prefix.
- Calibration: standardized L2 logistic regression fit on calibration partitions only; no test refitting.
- Evaluation: frozen held-out test partitions only.
- Descriptive endpoints: AUROC, AUPRC, AURC, and risk at 50% and 80% coverage.
- Aggregation: cell-level estimates followed by equal-cell macro aggregation where cells share the frozen outcome, split, metric, and aggregation contract.
- Uncertainty: paired, class-stratified within-cell bootstrap with 10,000 replicates and seed 2027.
- Presentation: fraction-versus-metric curves with confidence intervals. No post-hoc “earliest useful fraction” threshold is authorized.

## Fail-closed execution gates

Execution must stop unless all of the following hold:

1. RQ4 generation is exactly 19,420/19,420.
2. All 60 shard audits are present and PASS.
3. The generation `final_manifest.json` and its SHA-256 sidecar validate.
4. The frozen RQ4 downstream finalization and evidence gate have completed.
5. Every analyzed row contains the expected ordered telemetry and belongs to the registered calibration or test partition.
6. No feature reads observations after its fixed prefix boundary.
7. No partial RQ4 result was used to alter fractions, features, models, endpoints, or aggregation.

The analysis must remain unpublished and absent from manuscript result claims until those gates pass and a separate immutable result manifest authorizes it.

## RQ5 sample-budget disposition

The requested black-box K=2/3/5 curve is **not currently authorized under the all-eligible-example condition**. Terminal accounting contains 46,681 schema-valid responses among 58,260 registered requests, with malformed, provider-failure, and ambiguous/locked terminal categories. Therefore not every registered eligible example has five valid responses. Ambiguous calls must not be replayed, and a post-hoc complete-case subset must not silently replace the frozen population. No new API calls are authorized.
