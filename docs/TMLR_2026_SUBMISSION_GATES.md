# TMLR 2026 submission gates

Policy audit date: 2026-10-24

This checklist translates the current official TMLR policies into repository gates. It is an operational aid, not a substitute for rechecking the official pages immediately before submission.

## Official sources checked

- [Author Guide](https://www.jmlr.org/tmlr/author-guide.html)
- [Editorial Policies](https://jmlr.org/tmlr/editorial-policies.html)
- [Acceptance Criteria](https://jmlr.org/tmlr/acceptance-criteria.html)
- [Submission Instructions](https://jmlr.org/tmlr/submissions.html)
- [FAQ](https://www.jmlr.org/tmlr/faq.html)
- [Increasing emphasis on clear writing](https://blog.tmlr.org/2026/increasing-emphasis-on-clear-writing-in-acceptance-criteria/)
- [Asking authors about their own papers](https://blog.tmlr.org/2026/asking-authors-about-their-own-papers/)
- [October 2026 process update](https://blog.tmlr.org/2026/adapting-our-processes-to-the-changing-world-of-scientific-publication/)

## Hard submission gates

| Gate | Repository check | Current state |
|---|---|---|
| Claims are supported by accurate, convincing evidence | `claim_evidence_matrix.tsv`; generated result macros; final artifact validation | **BLOCKED** for unfinished RQ2/RQ4/RQ5; RQ1 authorized |
| Work is of interest and clearly communicated | title/abstract/section architecture audit; compiled-paper review | IN REVIEW |
| Double-blind anonymity | `python scripts/audit_paper.py --paper-dir paper` | AUTOMATED |
| Official TMLR template is unchanged | `paper/tmlr.sty`, `paper/main.tex`, paper audit, compile log | AUTOMATED + VISUAL |
| Main body length is editorially defensible | page count and density inspection; appendix after references | MANUAL AT FINAL PDF |
| Originality and concurrent-submission rules are satisfied | `TMLR_ORIGINALITY_AUDIT.md` plus author attestation | **AUTHOR ATTESTATION REQUIRED** |
| References are accurate and primary where possible | `TMLR_CITATION_FORENSICS.md`; citation-key audit | PARTIAL ONLINE CHECK; FINAL HUMAN CHECK REQUIRED |
| AI assistance is disclosed | first-page footnote and `AI-Assisted Tools` statement | PRESENT |
| Supplement is anonymous, nonessential, and within limits | deterministic supplement package and size audit | AUTOMATED AT RELEASE |
| No unfinished empirical claim reaches the submission PDF | `paper_evidence_pipeline.py`; submission-ready audit | **FAIL-CLOSED** |

## Acceptance-criterion translation

TMLR asks whether the paper's claims are supported and whether some readers would find the work interesting, with clear communication part of the latter criterion. For TrajUQ this means:

1. The primary claim is incremental and testable: trajectory features must improve over matched decision-time entropy, not merely correlate with errors.
2. The three observability regimes are never pooled into a causal architecture ranking.
3. Every answer paragraph names its evidence, uncertainty, gate, and boundary.
4. Heterogeneity, calibration regressions, and null results are retained rather than hidden.
5. Operational progress, GPU utilization, and partial cells never authorize a scientific claim.

## Pre-submission command gate

Run from a clean checkout:

```bash
python scripts/source_only_manuscript_audit.py
python scripts/audit_paper.py --paper-dir paper
python scripts/paper_evidence_pipeline.py --help
python scripts/build_overleaf_package.py --help
```

The final package must additionally pass the submission-ready evidence pipeline, deterministic two-build hash comparison, PDF compilation, log inspection, page-by-page visual review, package-content audit, and author originality attestation. A failed or unavailable check remains a blocker; it is never converted to PASS by prose.

