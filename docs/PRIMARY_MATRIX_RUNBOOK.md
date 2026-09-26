# Primary matrix runbook

The full primary matrix is intentionally impossible to plan until the integration
gates are complete.

## 1. Pass the integration readiness gate

Required artifacts:

- LLaDA exact parity PASS;
- SDAR exact parity PASS;
- Dream exact parity PASS;
- LLaDA + GSM8K 50-example pilot quality PASS;
- human pilot-review attestation PASS.

Then:

```bash
python scripts/primary_readiness.py
```

Do not continue unless `results/primary_readiness.json` says `PASS`.

## 2. Freeze exact model revisions without editing tracked templates

For each primary model, resolve an immutable Hugging Face SHA, then create an
artifact-local config.

Example:

```bash
python scripts/resolve_hf_revision.py \
  --repo-type model \
  --repo-id inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --output results/llada_model_revision.json

python scripts/freeze_model_config.py \
  --template configs/models/llada.yaml \
  --revision <EXACT_LLADA_SHA> \
  --output results/frozen_models/llada.yaml
```

Repeat for `sdar` and `dream`.

## 3. Freeze all dataset snapshots

Run `resolve_hf_revision.py --repo-type dataset` for every primary dataset and
prepare its pair with the exact SHA:

```bash
python scripts/prepare_eval_pair.py \
  --dataset <DATASET> \
  --revision <EXACT_DATASET_SHA>
```

The planner will reject a split manifest whose `requested_revision` is empty.
GPQA remains gated and must be accessed under its upstream terms.

## 4. Build, inspect, and archive the execution plan

```bash
python scripts/plan_primary_matrix.py
```

For the frozen 3-model × 6-dataset primary configuration, the plan contains 18
model--dataset cells and separate calibration/test generation, raw audit,
scoring, and scored-audit argv arrays for each cell.

The planner **does not execute GPU work**. This is deliberate: inspect and archive
`results/primary_matrix_plan.json` together with the freeze manifests before
launching expensive runs.

## 5. Execute without mutation

When executing the plan:

- use the argv values exactly;
- never overwrite completed raw runs;
- never regenerate data partitions after outcomes are seen;
- do not edit the feature dictionary, calibration family, or statistical rules
  because a test result is weak;
- preserve raw/scored artifacts separately;
- archive every audit, environment snapshot, and SHA-256 manifest.

After cell-level metrics are exported, use `scripts/aggregate_cells.py` for the
hierarchical equal-cell-weight macro analysis.


## 6. Adaptive second-pass plan

The full-budget primary outputs double as the refined-budget artifacts for RQ3.
After the primary execution plan is frozen, generate the separate low-budget
base-run plan:

```bash
python scripts/plan_adaptive_runs.py
```

The frozen budgets are:

| Model | Base | Refined |
|---|---:|---:|
| LLaDA | 32 steps | 128 steps |
| Dream | 32 steps | 128 steps |
| SDAR | 2 denoising steps/block | 4 denoising steps/block |

The primary selection cap is 50%; 25% and 75% are sensitivity settings. Selected
examples are evaluated under the conservative **second-pass rerun** semantics:
the base run is always paid and the refined run is an additional complete
generation. The planner never uses the cheaper continuation cost formula unless
a future runtime genuinely resumes the same denoising trajectory.
