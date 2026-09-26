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

The official acl.sty should come from the current ACL template rather than being vendored here.

## Result-population rule

The TBD cells are development placeholders only. Before submission, every placeholder must be replaced by a number generated from a frozen artifact, or the corresponding row/table must be removed. Never manually type an empirical value that cannot be regenerated from the repository pipeline.

## Main-page budget

Keep the main scientific content within 8 pages. Limitations, optional ethics, references, and appendices follow the current ARR rules and should remain in the same PDF. Run the official ACL pubcheck before submission.


## Static audit

During development:

    python scripts/audit_paper.py

Immediately before ARR submission:

    python scripts/audit_paper.py --submission-ready

Submission-ready mode fails if any TBD placeholder remains, in addition to the
anonymity and structural checks.
