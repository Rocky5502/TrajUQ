# RQ5 terminal reconciliation

RQ5 is authorized for analysis only after every frozen request UUID belongs to
exactly one terminal category: `valid`, `malformed`, `provider_failure`,
`ambiguous_locked`, or another explicitly registered terminal category.
`missing_unreconciled` is never terminal and therefore fails the gate.

The reconciler binds the freeze hash, partition hashes and row counts, model,
dataset, split, example identifier, repeat index, and deterministic request
UUID. It validates record/error provenance without contacting the provider.
An ambiguous request is permanently locked against replay. Its pending marker
and any associated error record are hashed into the terminal ledger even when
the older operator registry did not enumerate that UUID.

Read-only audit:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/reconcile_rq5_terminal_registry.py --check-only
```

Materialization must use new, isolated paths. Existing response registries,
manifests, provider records, errors, and pending locks must never be overwritten:

```bash
python scripts/reconcile_rq5_terminal_registry.py \
  --ledger-output results/rq5/reconciliation/terminal_ledger.jsonl \
  --manifest-output results/rq5/reconciliation/terminal_manifest.json
```

The terminal manifest deliberately sets `analysis_authorized=false`.
Reconciliation establishes exact request accounting; it does not replace the
registered scoring, calibration-only fitting, held-out evaluation, bootstrap,
multiplicity correction, or final RQ5 claim gate.
