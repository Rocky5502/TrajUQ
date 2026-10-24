# TMLR citation forensics

Audit date: 2026-10-24

## Automated findings

- Manuscript citation keys: 33
- Bibliography entries: 33
- Missing cited keys: 0
- Duplicate bibliography keys: 0
- Structural audit: PASS
- Resolver inventory: 11 DOI records, 11 explicit URLs, and 12 additional arXiv identifiers were extracted.

Network reachability was checked without changing the bibliography. Several primary pages and DOI resolvers returned success or redirect responses; several arXiv/DOI requests timed out from the audit host. A timeout is recorded as inconclusive, not as evidence that a paper is absent.

## High-risk recent-citation review

| Key | Why high risk | Required primary record | Status |
|---|---|---|---|
| `cheng2026sdar` | recent proceedings metadata | ACL Anthology DOI record | primary DOI present; final metadata check required |
| `zhu2026flare` | recent arXiv work | arXiv record 2606.01774 | identifier present; resolver inconclusive on audit host |
| `zhong2026dise` | close scientific precedent | ACL Anthology 2026.acl-long.298 | primary page and DOI present; page reachable |
| `vazhentsev2026dllmuq` | closest systematic dLLM-UQ precedent | arXiv record 2605.14570 | identifier and URL present; resolver inconclusive on audit host |
| `yadav2026unsure` | close robustness precedent | arXiv record 2608.08791 | identifier and URL present; resolver inconclusive on audit host |
| `chen2026stdec` | recent trajectory-decoding work | arXiv record 2604.06330 | identifier present; resolver inconclusive on audit host |
| `lu2026pacedllm` | recent trajectory-decoding work | arXiv record 2609.26249 | identifier present; resolver inconclusive on audit host |
| `siliconflow*` | mutable provider documentation/pricing | captured provider pages plus frozen pricing contract | URLs present; reasoning/pricing pages reachable; chat endpoint inconclusive |

## Claim-to-citation boundaries

1. Related-work citations support descriptions of prior methods, not TrajUQ's empirical findings.
2. The manuscript explicitly says trajectory-based dLLM uncertainty is not newly invented here.
3. Provider documentation supports only interface/request descriptions; it does not support claims about hidden reasoning, weights, routing, hardware, or precision.
4. “Competitive,” “systematic,” and similar adjectives must remain attributable to the cited source and should be removed if the primary record does not substantiate them.
5. Final submission requires a human metadata pass for all recent 2025--2026 records whose resolver check was inconclusive.

## Decision

`PASS` for source-level citation integrity; `BLOCKED` for final-submission certification until the inconclusive recent records receive a human primary-source check. This distinction prevents a network timeout from being silently converted into bibliographic certainty.

