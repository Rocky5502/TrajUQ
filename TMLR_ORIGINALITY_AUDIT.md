# TMLR originality audit

Audit date: 2026-10-24

## Repository-side finding

No manuscript file in this checkout is identified as a verbatim import from a prior archival publication. The cited nearest work is treated as related work rather than as evidence for TrajUQ's empirical claims. The central novelty claim is deliberately narrow: an incremental, held-out, matched-baseline test of lightweight native-generation telemetry, extended through a prospectively frozen observability ladder.

## Nearest-work boundaries

| Work | Overlap | TrajUQ boundary |
|---|---|---|
| DiSE | dLLM confidence/self-evaluation via regeneration | TrajUQ studies native-generation telemetry and a matched decision-time baseline rather than regeneration as the principal estimator |
| Vazhentsev et al. dLLM-UQ | trajectory entropy, likelihood, flips, remasking, semantic instability | TrajUQ does not claim invention of trajectory UQ; it tests incremental value under a frozen active-unresolved schema, three families, adaptive compute, and an observability ladder |
| Yadav et al. | representation--confidence gap | TrajUQ uses no hidden-state predictor and makes no representation-level claim |
| Semantic entropy / repeated sampling | sample disagreement for LM uncertainty | TrajUQ treats repeated sampling as a costed comparator, not as its native single-run signal |
| STDec / PACE-dLLM / dynamic-threshold decoding | confidence trajectories for efficient decoding | TrajUQ's outcome is answer-level error ranking and calibrated selection; decoding policy is not redesigned after outcomes |

## Text, figure, result, and submission checks

- [x] Manuscript cites the nearest scientific precedents and states a narrower novelty boundary.
- [x] Generated figures/tables are tied to repository renderers and artifact manifests.
- [x] The review PDF is anonymous and contains no public repository URL.
- [ ] Authors confirm that no text, figure, table, or empirical result is reused from a published, accepted, or concurrently submitted archival paper in a way prohibited by TMLR.
- [ ] Authors list every non-archival workshop paper, thesis chapter, or preprint with material overlap and explain the delta in the submission form.
- [ ] Authors confirm that this manuscript is not under review at another archival venue.
- [ ] Authors confirm compliance with the current TMLR authorship quota and author-list policy.

## Decision

`POTENTIAL_CONFLICT / AUTHOR ATTESTATION REQUIRED`.

Repository evidence supports a distinct contribution boundary, but publication history and concurrent submissions cannot be inferred from the checkout. Submission is blocked until the authors complete the four unchecked attestations above.

