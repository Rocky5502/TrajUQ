# Experiment protocol

This is the reviewer-facing index to the frozen TrajUQ protocol. The authoritative executable contracts remain the versioned configs, schemas, and scripts.

## Immutable experiment identity

- Experiment commit: `28002055c828c6910840de6f2105aa6367514144`
- Primary config: `configs/experiments/primary.yaml`
- Model configs: `configs/models/`
- Dataset/split contract: `data/manifests/evaluation_splits.json`
- Analysis plan: `docs/ANALYSIS_PLAN.md`
- Evidence map: `docs/EVIDENCE_MAP.md`

## Stage contract

1. Resolve exact model/tokenizer and dataset revisions.
2. Validate runtime and instrumented/uninstrumented parity.
3. Freeze outcome-independent partitions and caps.
4. Generate raw pre-outcome records without references or labels.
5. Audit schema, provenance, exact IDs, uniqueness, and hashes.
6. Score immutable generations in a separate stage.
7. Fit calibrators on calibration partitions only.
8. Evaluate paired held-out examples under the registered aggregation/inference plan.
9. Evaluate the registered claim gate and authorize only complete evidence.
10. Export evidence/reproducibility packages from the authorized artifacts.

## Evidence-state contract

```text
planned
→ generated
→ saved
→ audited
→ scored
→ statistically complete
→ gate evaluated
→ claim authorized
→ released evidence
```

No later state may be inferred from an earlier state.

## Non-negotiable scientific invariants

- No change to prompts, seeds, decoding, revisions, precision, telemetry, or statistical gates after outcome inspection.
- No selective rerun of valid completed examples.
- No merge of a scale-out artifact without explicit ownership and exact-complete validation.
- No claim from partial progress, GPU utilization, or un-audited output.
- No silent replacement of exact model/data revisions with moving upstream heads.
- Negative, null, or heterogeneous outcomes remain valid scientific results.

## Repository boundary

This repository ends at research evidence and reproducibility packaging. Manuscript/venue authoring source is intentionally maintained outside the repository.

See `docs/PROTOCOL.md`, `docs/DATA_SPLITS.md`, `docs/REPRODUCIBILITY.md`, `docs/EVIDENCE_MAP.md`, and the adapter-specific instrumentation documents for full detail.
