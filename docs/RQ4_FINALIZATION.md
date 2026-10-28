# RQ4 fail-closed finalization

RQ4 downstream analysis is isolated from generation and cannot begin until the
sampled-control campaign has exactly **19,420/19,420** rows in **60/60**
complete, audited shards. `scripts/finalize_rq4_evidence.py` validates only
manifest metadata and immutable file hashes during this readiness gate; it does
not read partial outcomes.

The separate downstream plan must be frozen with `status: PASS`, bind the exact
final generation-manifest hash and every analysis script hash, and retain the
registered statistics (10,000 paired class-stratified bootstrap replicates,
seed 2027, equal cell weights, and Holm--Bonferroni correction over H4a--H4c).
The orchestrator then enforces this order:

1. scoring;
2. feature extraction, including the registered five-sample disagreement;
3. calibration-only fitting;
4. held-out evaluation;
5. paired bootstrap;
6. multiplicity correction;
7. hypothesis gate;
8. equal-cell macro aggregation.

Generation commands are forbidden in the downstream plan. Every stage owns
non-overlapping outputs, refuses unregistered overwrites, and is hash-checked
again on resume. The generation checkout, addendum, qualification, final
generation manifest, and all 60 owned raw artifacts are revalidated before
every stage. Any mismatch blocks the chain.

The resulting RQ4 evidence manifest remains manuscript independent and records
`claim_authorized: false`: mechanical completeness is not itself a positive or
negative scientific result. Scientific wording must follow the separately
registered hypothesis-gate artifact, including valid negative results.

No production plan should be frozen from partial outputs. Once the full
campaign audit writes its immutable final manifest, freeze the downstream plan
without inspecting outcomes, then run:

```bash
python scripts/finalize_rq4_evidence.py \
  --root /path/to/clean/analysis-checkout \
  --generation-root /data/TrajUQ_RQ4_sampling_5a3993c \
  --generation-manifest /data/trajuq_worker_state/a100-01/rq4_full_5a3993c/final_manifest.json \
  --plan /path/to/frozen/rq4_downstream_plan.json
```
