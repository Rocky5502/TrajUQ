# RQ2 evidence finalization

RQ2 finalization is deliberately fail closed and manuscript independent. The
canonical generation worker remains authoritative until all 120 seed jobs and
all 24 generation partitions pass their frozen audits.

Do not run the finalizer during generation. A readiness check against an
incomplete campaign exits `BLOCKED` before executing scoring or aggregation.

After canonical generation completes, run from a clean checkout of the
long-term public repository. Point it at the unchanged authoritative experiment
checkout at commit `28002055c828c6910840de6f2105aa6367514144`:

```bash
python scripts/finalize_rq2_evidence.py --root /data/TrajUQ
```

The command performs the registered chain:

1. validate the frozen 12-cell plan and provenance;
2. require exact ID coverage and PASS raw audits for 120 seed jobs;
3. run repository-supported scoring, pair evaluation, comparison, and macro
   aggregation with verified artifact reuse;
4. require 12 complete cells and the frozen macro protocol;
5. recount 120/120 seed gates and 24/24 partitions from the execution snapshot;
6. write `results/generated/final/rq2_manifest.json` atomically;
7. validate the candidate manifest against the authoritative commit and counts.

The command never edits `ARTIFACT_MANIFEST.json`, never changes manuscript
claims, and reports `claim_authorized: false`. Scientific authorization is a
separate review step after the generated manifest, registered comparisons,
compute ledger, and reproducibility checks have all been inspected.

The exact commands can be previewed only after all generation gates pass:

```bash
python scripts/finalize_rq2_evidence.py --root /data/TrajUQ --dry-run
```

Partial progress summaries, throughput estimates, and operational dashboards
are never substitutes for the final manifest.
