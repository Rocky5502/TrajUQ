# TrajUQ TMLR manuscript

This directory is an anonymous **pre-results** long-paper source.

Current submission target:
- the official TMLR style in anonymous review mode;
- double-blind review through OpenReview;
- length justified by content (TMLR has no fixed page limit);
- a Statement of Broader Impact when the work carries significant risk of harm;
- references followed by the optional appendix in the same PDF;
- anonymized supplementary material, up to 100 MB, in PDF or ZIP format.

Do not add author names, acknowledgements, the public GitHub URL, institution-revealing paths, or result numbers during anonymous review.

## Build

Start from the current official TMLR Overleaf template and replace its main.tex with this file, then add:
- appendix.tex
- references.bib
- generated_results.tex

The official TMLR style is not vendored in the repository. Build a deterministic
anonymous Overleaf ZIP from the pinned official style commit with:

    python scripts/build_overleaf_package.py

The package contains `main.tex`, `appendix.tex`, `references.bib`,
`generated_results.tex`, the pinned official `tmlr.sty`, `tmlr.bst`, and
`fancyhdr.sty`, generated result PDFs when present, and an internal SHA-256
manifest. Packaging requires a clean checkout.

## Result-population rule

The committed `generated_results.tex` is a pre-results fallback containing TBD macros. Never manually type empirical values into `main.tex`. After the complete primary, RQ1 claim-gate, 18-cell feature-ablation, frozen 12-cell repeated-sampling subset, and 50% adaptive macro artifacts exist, run:

    python scripts/render_results_tex.py

The renderer refuses incomplete cell sets, cross-checks the RQ1 claim gate against the primary macro result, requires the complete hierarchical feature-ablation and prior-aligned diagnostic macros, requires the complete 18-cell supplementary isotonic calibration-sensitivity PASS/SKIP partition, requires repeated-sampling macro protocol v2 with complete five-sample auxiliary wall-clock accounting across the frozen config-bound 12-cell subset, writes all table/cost/ablation/calibration-sensitivity macros, and emits `paper/generated_results_manifest.json` protocol v2 with SHA-256 bindings to eight sources: primary, RQ1 gate, repeated sampling, feature ablations, prior-aligned signal diagnostics, supplementary isotonic calibration sensitivity, adaptive compute, and the frozen primary config. A failed RQ1 gate produces narrow result language rather than a positive claim, and the isotonic sensitivity never changes the frozen RQ1 decision.

After all three adaptive-cap macros exist, also run:

    python scripts/render_paper_result_figures.py

This writes three publication-ready vector PDFs plus `paper/figures/generated/result_figure_manifest.json` protocol v2. The manifest binds the primary/config/adaptive sources and each PDF hash. The primary 18-cell AUROC matrix is conditionally included in the main Results section; the family-level AUROC and adaptive accuracy--cost figures remain in the appendix. The renderer also emits a complete artifact-derived 18-cell descriptive table through `generated_results.tex`. Final submission readiness rejects stale or missing figure/table evidence.

## Length and review format

TMLR imposes no fixed page limit, but length must be justified by content and unusually long submissions may face review delays. Keep the paper concise and complete, preserve double-blind anonymity, and compile the exact final ZIP with the pinned official TMLR template before submission.


## Static audit

During development (this also checks citation-key integrity, duplicate labels, and missing LaTeX references):

    python scripts/audit_paper.py

Immediately before TMLR submission:

    python scripts/audit_paper.py --submission-ready

Submission-ready mode fails if any TBD placeholder remains, in addition to the
anonymity and structural checks.


## TMLR target

The target is the rolling Transactions on Machine Learning Research review
process through OpenReview. See `docs/TMLR_EXTENSION.md` for the prospective
RQ4 boundary and venue requirements. The development repository must not be
linked from the anonymous review PDF.


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

The remaining external checks are compiling the exact recorded Overleaf ZIP
with the pinned official TMLR template, inspecting anonymity and OpenReview
metadata, and uploading the exact archives whose hashes are recorded in
`results/submission_readiness.json`.
