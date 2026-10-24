# Evidence map

This document maps TrajUQ's scientific stages to inspectable artifacts. It intentionally describes **research evidence**, not manuscript placement.

## Evidence flow

~~~text
frozen protocol
→ generation
→ immutable raw artifact
→ provenance/schema audit
→ separate correctness scoring
→ calibration
→ held-out evaluation
→ registered aggregate/statistic
→ claim gate
→ authorized evidence
~~~

No later state is inferred from an earlier state.

## Released authorized evidence

The machine-readable authority is \`ARTIFACT_MANIFEST.json\`.

| Artifact | Role |
|---|---|
| \`results/primary_analysis/macro_primary.json\` | complete RQ1 primary aggregate |
| \`results/primary_analysis/rq1_claim_gate.json\` | frozen RQ1 claim decision |
| \`results/primary_analysis/macro_feature_ablation.json\` | complete feature-ablation aggregate |
| \`results/primary_analysis/macro_prior_signal_diagnostics.json\` | registered prior-aligned signal diagnostics |
| \`results/primary_analysis/macro_calibration_sensitivity.json\` | supplementary calibration sensitivity |
| \`results/compute_budget/environment_manifest.json\` | authoritative runtime/host snapshot |

Each released item is SHA-256-bound by the manifest.

## Pre-experiment / execution gates

Full reproduction may additionally create:

- \`results/cpu_verification.json\`
- \`results/revision_ledger.json\`
- \`results/gated_dataset_access.json\`
- \`results/dataset_contracts.json\`
- \`results/llada_parity.json\`
- \`results/sdar_parity.json\`
- \`results/dream_parity.json\`
- \`results/llada_gsm8k_pilot50_gate.json\`
- \`results/llada_gsm8k_pilot50_manual_review.json\`
- \`results/math500_scorer_audit.json\`
- \`results/math500_scorer_manual_review.json\`
- \`results/primary_readiness.json\`

These authorize execution; they are not automatically scientific outcome claims.

## RQ1

Primary cell-level evidence lives under:

~~~text
results/primary_analysis/<adapter>/<dataset>/
~~~

The complete cross-cell artifacts are the released macro files listed above. RQ1 uses the frozen three-model × six-dataset matrix and its registered equal-cell aggregation/gate.

## RQ2

Registered final aggregate:

\`results/repeated_sampling/macro_trajuq_vs_repeated.json\`

The frozen repeated-sampling comparator covers three primary diffusion models × GSM8K, MATH-500, GPQA-Diamond, and MMLU-Pro with five seeds (2027–2031). ARC-Challenge and TruthfulQA are not part of this compute-bounded comparator.

RQ2 is not authorized in a release until the registered complete aggregate and its provenance gates pass.

## RQ3

Registered adaptive-compute aggregates:

- \`results/adaptive/macro_cap_025.json\`
- \`results/adaptive/macro_cap_050.json\` — primary cap
- \`results/adaptive/macro_cap_075.json\`

The registered semantics are selective second-pass reruns with calibration-only policy selection.

## RQ4

RQ4 is a separate autoregressive-control extension. Its frozen configuration is:

\`configs/experiments/tmlr_rq4.yaml\`

Its evidence must remain separate from the original RQ1 diffusion-family gate.

## RQ5

RQ5 black-box configuration:

\`configs/experiments/tmlr_rq5_blackbox.yaml\`

Provider-visible model IDs:
- \`Pro/deepseek-ai/DeepSeek-V3.2\`
- \`Pro/zai-org/GLM-5.1\`
- \`Pro/moonshotai/Kimi-K2.6\`

Black-box results are authorized only from validated provider-visible responses. Hidden weights, backend routing, and provider hardware are outside the observable evidence boundary.

## Validate the released evidence

~~~bash
python reproduction/reproduce_core.py
~~~

For full CPU software checks:

~~~bash
make verify
~~~

## Pending science

\`ARTIFACT_MANIFEST.json\` explicitly records pending scientific components. Pending entries are never substituted with estimates, partial-run summaries, or operational progress.
