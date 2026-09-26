# TrajUQ ARR / ACL manuscript

This directory is an anonymous **pre-results** long-paper source.

Current ARR formatting target:
- official ACL style with review mode;
- at most 8 pages of main content for a long paper;
- a mandatory **Limitations** section after the conclusion;
- optional **Ethical Considerations**;
- references and appendices in the same PDF;
- appendices remain double-column unless a math-heavy subsection genuinely requires otherwise.

Do not add author names, acknowledgements, the public GitHub URL, institution-revealing paths, or result numbers during anonymous review.

## Build

Start from the current official ACL Overleaf template and replace its main.tex with this file, then add:
- appendix.tex
- references.bib
- generated_results.tex

The official acl.sty should come from the current ACL template rather than being vendored here.

## Result-population rule

The committed `generated_results.tex` is a pre-results fallback containing TBD macros. Never manually type empirical values into `main.tex`. After the complete primary, repeated-sampling, RQ1 claim-gate, and 50% adaptive macro artifacts exist, run:

    python scripts/render_results_tex.py

The renderer refuses incomplete cell sets, cross-checks the RQ1 claim gate against the primary macro result, writes all table macros, and emits `paper/generated_results_manifest.json` with source SHA-256 hashes. A failed RQ1 gate produces narrow result language rather than a positive claim.

## Main-page budget

Keep the main scientific content within 8 pages. Limitations, optional ethics, references, and appendices follow the current ARR rules and should remain in the same PDF. Run the official ACL pubcheck before submission.


## Static audit

During development (this also checks citation-key integrity, duplicate labels, and missing LaTeX references):

    python scripts/audit_paper.py

Immediately before ARR submission:

    python scripts/audit_paper.py --submission-ready

Submission-ready mode fails if any TBD placeholder remains, in addition to the
anonymity and structural checks.


## NAACL 2027 target

The target is the ARR October 2026 cycle for NAACL 2027. The current official
ARR deadline is **2026-10-12 (AoE)**. See `docs/NAACL2027_SUBMISSION.md` for the
verified submission/anonymity/service checklist. The development repository
must not be linked from the anonymous review PDF.
