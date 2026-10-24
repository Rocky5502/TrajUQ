# Compute accounting

TrajUQ reports result-producing compute from artifact timestamps and hashed manifests.

## Definitions

- **GPU-hours:** result-producing wall-clock hours multiplied by physical GPU count.
- **Utilization:** a separate monitoring signal; never a multiplier for GPU-hours.
- **Auxiliary sampling cost:** measured independently and reported with the repeated-sampling comparison.
- **Hosted API cost:** provider-reported input/output tokens evaluated against a captured pricing contract.
- **Diagnostic/development compute:** reported separately when reconstructable; never folded into an invented precise total.

## Authoritative sources

- `results/compute_budget/environment_manifest.json`
- `results/compute_budget/compute_ledger.jsonl`
- `results/compute_budget/compute_summary.json`
- `results/compute_budget/compute_budget_manifest.json`
- `paper/generated_compute.tex`

The manuscript renderer rejects unbound totals. Provider-side hardware, GPU count, routing, precision, and kernel choices for black-box RQ5 are unobservable and must not be inferred.

