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

The official ACL style is not vendored in the repository. Build a deterministic
anonymous Overleaf ZIP from the pinned official style commit with:

    python scripts/build_overleaf_package.py

The package contains `main.tex`, `appendix.tex`, `references.bib`,
`generated_results.tex`, the pinned official `acl.sty` /
`acl_natbib.bst`, generated result PDFs when present, and an internal SHA-256
manifest. Packaging requires a clean checkout.

## Result-population rule

The committed `generated_results.tex` is a pre-results fallback containing TBD macros. Never manually type empirical values into `main.tex`. After the complete primary, RQ1 claim-gate, 18-cell feature-ablation, repeated-sampling, and 50% adaptive macro artifacts exist, run:

    python scripts/render_results_tex.py

The renderer refuses incomplete cell sets, cross-checks the RQ1 claim gate against the primary macro result, requires the complete hierarchical feature-ablation and prior-aligned diagnostic macros, requires the complete 18-cell supplementary isotonic calibration-sensitivity PASS/SKIP partition, requires repeated-sampling macro protocol v2 with complete five-sample auxiliary wall-clock accounting across all 18 cells, writes all table/cost/ablation/calibration-sensitivity macros, and emits `paper/generated_results_manifest.json` protocol v2 with SHA-256 bindings to eight sources: primary, RQ1 gate, repeated sampling, feature ablations, prior-aligned signal diagnostics, supplementary isotonic calibration sensitivity, adaptive compute, and the frozen primary config. A failed RQ1 gate produces narrow result language rather than a positive claim, and the isotonic sensitivity never changes the frozen RQ1 decision.

After all three adaptive-cap macros exist, also run:

    python scripts/render_paper_result_figures.py

This writes three vector PDFs plus `paper/figures/generated/result_figure_manifest.json` protocol v2. The manifest binds the primary/config/adaptive sources and each PDF hash; the appendix includes those figures conditionally, and final submission readiness rejects stale or missing figure evidence.

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


## Final repository-side submission gate

After all frozen results are rendered, the final repository gate builds and
records both anonymous upload archives:

    python scripts/submission_readiness.py

The gate requires every **required** project execution gate to be PASS, verifies
that the result-render and result-figure manifests still hash to their exact
source artifacts/outputs, runs the manuscript audit in submission-ready mode,
builds the Overleaf source ZIP with:

    python scripts/build_overleaf_package.py --submission-ready

and rebuilds the deterministic anonymous software supplement. The submission
JSON records SHA-256 values for **both** archives. Submission-ready Overleaf
packaging includes only the three result PDFs named by the validated figure
manifest; stale or unlisted generated PDFs are excluded.

The remaining external checks are compiling the exact recorded Overleaf ZIP,
running ACL pubcheck, inspecting page/anonymity constraints, and uploading the
exact archives whose hashes are recorded in `results/submission_readiness.json`.
