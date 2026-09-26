# Result artifact to manuscript map

This file is the canonical bridge between frozen TrajUQ experiment artifacts
and the anonymous ARR/NAACL manuscript. It does not contain empirical values.
Its purpose is to prevent manual copying, selective reporting, or uncertainty
about where a result belongs after the A100 runs complete.

The paper remains pre-results until every required artifact below passes its
own semantic/provenance gate.

## Pre-primary evidence

These artifacts authorize execution but do not directly populate paper result
numbers.

| Artifact | Producer / validator | Role |
|---|---|---|
| `results/cpu_verification.json` | `scripts/freeze_preprimary.sh` / `scripts/verify_cpu.sh` | clean CPU/static verification bound to the frozen experiment commit |
| `results/revision_ledger.json` | `scripts/resolve_experiment_revisions.py` | immutable model/data/config revision ledger |
| `results/llada_parity.json` | `scripts/pilot_linux.sh parity` | exact 20-prompt LLaDA parity-v3 evidence |
| `results/sdar_parity.json` | `scripts/parity_remaining_linux.sh sdar` | exact 20-prompt SDAR parity-v3 evidence |
| `results/dream_parity.json` | `scripts/parity_remaining_linux.sh dream` | exact 20-prompt Dream parity-v3 evidence |
| `results/llada_gsm8k_pilot50_gate.json` | `scripts/pilot_linux.sh pilot` | 50-example instrumentation/quality gate |
| `results/llada_gsm8k_pilot50_manual_review.json` | `scripts/record_manual_gate.py` | human review of frozen pilot traces |
| `results/math500_scorer_audit.json` | `scripts/audit_math500_scorer.py` | pinned automatic MATH-500 scorer audit |
| `results/math500_scorer_manual_review.json` | `scripts/record_math500_scorer_review.py` | 25-example calibration-only human scorer audit |
| `results/primary_readiness.json` | `scripts/primary_readiness.py` | hard authorization gate for the 18-cell primary matrix |

## Primary RQ1 artifacts

| Artifact | Protocol / completeness contract | Manuscript destination |
|---|---|---|
| `results/primary_analysis/macro_primary.json` | `hierarchical_within_cell_macro_bootstrap_v1`; complete 18-cell set | main Table `tab:primary`; main result figures; abstract/results/conclusion wording through generated macros |
| `results/primary_analysis/rq1_claim_gate.json` | `trajuq_rq1_claim_gate_v1`; frozen decision must match the primary macro/config | abstract, Results narrative, Conclusion narrative; controls whether a positive RQ1 claim is permitted |
| `results/primary_analysis/<adapter>/<dataset>/metrics.json` | `trajuq_heldout_evaluation_v1` | appendix per-cell analyses / diagnostic inspection |
| `results/primary_analysis/<adapter>/<dataset>/predictions.jsonl` | exact frozen test-ID coverage | input to complete primary macro and RQ2 comparison |
| `results/primary_analysis/<adapter>/<dataset>/nested_bootstrap.json` | `trajuq_nested_calibrated_comparison_v1` | per-cell paired RQ1 uncertainty analysis |
| `results/primary_analysis/<adapter>/<dataset>/raw_risk_diagnostic.json` | `trajuq_raw_risk_diagnostic_v1` | training-free sensitivity / diagnostic discussion only |

Primary manuscript macros are emitted by `scripts/render_results_tex.py`;
numbers must not be typed manually into `paper/main.tex`.

## Frozen feature-family ablations

Every cell produces both:

- `results/primary_analysis/<adapter>/<dataset>/feature_ablation.json`
- `results/primary_analysis/<adapter>/<dataset>/feature_ablation_predictions.jsonl`

The per-example prediction artifact must exactly cover the frozen test IDs and
contain calibrated probabilities for the exact frozen group set:

1. decision-only;
2. decision + distribution;
3. decision + stability;
4. decision + persistence;
5. full;
6. full minus JS drift;
7. full minus revisions;
8. full minus persistence.

The cross-cell output is:

`results/primary_analysis/macro_feature_ablation.json`

Required protocol: `trajuq_feature_ablation_macro_v1`.

It uses equal cell weight and the same paired within-cell hierarchical bootstrap
logic as the primary comparison. Selective cell omission is forbidden.

Manuscript destination:

- Appendix Table `tab:feature-ablation`;
- artifact-derived `Ablation*` LaTeX macros in
  `paper/generated_results.tex`.

## RQ2 five-sample comparator

Primary artifact:

`results/repeated_sampling/macro_trajuq_vs_repeated.json`

Required protocol: `trajuq_vs_repeated_macro_v2`.

Required properties:

- complete 18-cell set;
- exactly five auxiliary samples per cell;
- seeds 2027--2031;
- complete test-time auxiliary wall-clock accounting in all cells;
- equal-cell-weight macro;
- MATH-500 vote classes use reference-free pinned symbolic equivalence.

Manuscript destination:

- main Table `tab:repeated`;
- `Repeated*` metric macros;
- `RepeatedAuxWallTime` compute macro.

The target generation itself is excluded from the reported auxiliary repeated
generation cost, as recorded in the artifact definition.

## RQ3 adaptive compute

Primary artifact:

`results/adaptive/macro_cap_050.json`

Required protocol: `trajuq_adaptive_macro_v1`.

Required semantics:

- `selective_second_pass_rerun`;
- complete 18-cell set;
- 50% cap marked primary;
- calibration-only cross-fitted threshold selection;
- base, fixed-refined, and matched-random comparisons.

Sensitivity artifacts:

- `results/adaptive/macro_cap_025.json`;
- `results/adaptive/macro_cap_075.json`.

Manuscript destination:

- main Table `tab:adaptive` for the 50% primary cap;
- generated adaptive accuracy/cost macros;
- generated vector figure
  `paper/figures/generated/adaptive_accuracy_cost_tradeoff.pdf` for the
  25/50/75% sensitivity curve.

## Generated result figures

`scripts/render_paper_result_figures.py` consumes only complete frozen
artifacts and writes:

- `paper/figures/generated/primary_auroc_improvement_matrix.pdf`;
- `paper/figures/generated/family_macro_auroc_improvement.pdf`;
- `paper/figures/generated/adaptive_accuracy_cost_tradeoff.pdf`;
- `paper/figures/generated/result_figure_manifest.json`.

Required protocol: `trajuq_paper_result_figures_v2`.

The figure manifest SHA-binds the primary aggregate, frozen primary config, all
three adaptive-cap artifacts, and each generated PDF by SHA-256 and byte count.
`scripts/submission_readiness.py` revalidates those bindings before the final
repository-side submission gate can pass.

## Final paper render

`scripts/render_results_tex.py` consumes six SHA-bound sources:

1. `results/primary_analysis/macro_primary.json`;
2. `results/primary_analysis/rq1_claim_gate.json`;
3. `results/repeated_sampling/macro_trajuq_vs_repeated.json`;
4. `results/primary_analysis/macro_feature_ablation.json`;
5. `results/adaptive/macro_cap_050.json`;
6. `configs/experiments/primary.yaml`.

It writes:

- `paper/generated_results.tex`;
- `paper/generated_results_manifest.json`.

`scripts/submission_readiness.py` deterministically re-renders these macros
from the bound sources and rejects manual TeX/manifest tampering.

## Optional RQ4 diagnostic

`results/flare_rq4_hybrid.json` is optional and must remain explicitly labeled
`RQ4_DIAGNOSTIC_NOT_PRIMARY_ENDPOINT`.

It may support a diagnostic subsection/appendix if the same-checkpoint
AR/diffusion parity and trace gates pass. It must not modify or replace the
primary RQ1 claim gate.

## Final order after GPU execution

1. Require all pre-primary gates to PASS.
2. Complete and audit all 18 primary cells.
3. Complete primary analysis, including the 18-cell feature-ablation macro.
4. Freeze the RQ1 claim decision unchanged.
5. Complete RQ2 with compute accounting.
6. Complete RQ3 at 25/50/75% caps.
7. Run `scripts/render_results_tex.py`.
8. Run `scripts/render_paper_result_figures.py`.
9. Run `scripts/audit_paper.py --submission-ready`.
10. Commit/stash paper-side edits so the submission checkout is clean.
11. Run `scripts/submission_readiness.py`.
12. Only then perform the external ACL/ARR compile, pubcheck, anonymity,
    metadata, and upload checks.

No result may be entered into the manuscript without a traceable frozen
artifact in this map.
