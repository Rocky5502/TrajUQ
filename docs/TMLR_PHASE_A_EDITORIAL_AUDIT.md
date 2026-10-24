# TMLR Phase-A editorial audit

Audit date: 2026-10-24  
Source manuscript: `paper/main.tex`  
Scope: editorial structure and evidence discipline; no frozen scientific setting was changed.

## One-sentence core

TrajUQ asks what uncertainty remains useful as observability decreases from native diffusion trajectories, through an open autoregressive trace, to hosted black-box behavior.

## Title audit

Candidates:

1. **What Uncertainty Survives Reduced Observability? Trajectory-Aware Risk Estimation in Language Generation**
2. **What Uncertainty Remains Useful as Observability Decreases?**
3. **Trajectory-Aware Uncertainty Beyond Decision-Time Confidence**

Candidate 1 is selected. It states the journal-level scientific question, names the technical contribution, avoids implying that unfinished cross-regime experiments already support a positive conclusion, and renders cleanly in the official template. Candidate 2 is less specific about the technical contribution; candidate 3 understates the registered observability ladder.

## Paragraph-level classification

| Manuscript region | Primary role | Editorial action |
|---|---|---|
| Abstract | Problem, gap, method, protocol, authorized result, boundary | Rewritten to a seven-part claim skeleton; unfinished downstream claims remain gated |
| Introduction ¶1--2 | Problem and gap | Retained and tightened around the observability question |
| Introduction ¶3 | Method contract | Retained; emphasizes pre-outcome isolation |
| RQ list | Registered questions | Retained; no outcome-dependent revision |
| Contribution list | Contributions | Compressed; evidence pipeline separated from scientific contribution |
| Related work | Boundary and comparison | Organized by diffusion LMs, dLLM UQ, LM UQ, and selective prediction |
| Problem setup and method | Technical definition | Retained; scope and leakage boundaries made explicit |
| Experimental design | Protocol | Added a three-regime observability ladder and noncausal comparison boundary |
| Registered analysis contract | Statistical contract | Moved before Results; negative/null outcomes explicitly retained |
| Results RQ1--RQ5 | Question, design, evidence/answer, boundary | Re-architected into a consistent per-RQ pattern |
| Cross-regime synthesis | Descriptive synthesis | Explicitly prohibits raw cross-protocol ranking and causal interpretation |
| Conclusion | Bounded takeaway | Retains the artifact-generated result sentence and falsifiability |
| Limitations | Construct/internal/external/statistical/reproducibility validity | Expanded and structured |
| AI disclosure | Tool-use transparency | First-page disclosure preserved; detailed statement added |

## Reviewer panel

### Reviewer 1: statistical rigor

Primary concern: many cells, metrics, and extensions can invite selective interpretation. Resolution: the manuscript foregrounds one confirmatory RQ1 endpoint, paired within-cell bootstrap, equal-cell aggregation, a family-direction gate, and registered analysis contracts. Remaining concern: final RQ2--RQ5 multiplicity outputs must be checked from their immutable registries.

### Reviewer 2: diffusion-language-model specialist

Primary concern: trajectory-based dLLM uncertainty is not itself new. Resolution: novelty is framed as an incremental, matched-baseline test; the paper cites DiSE and recent systematic dLLM-UQ work and does not claim invention of trajectory uncertainty. Remaining concern: parity evidence and telemetry semantics must stay prominent in the supplement.

### Reviewer 3: general ML/TMLR

Primary concern: the five-RQ scope could read as several papers. Resolution: all questions now answer one observability ladder, and each regime has a bounded claim. Remaining concern: final main-body length and figure density require PDF-level review.

### Reviewer 4: reproducibility

Primary concern: a large artifact graph could make the paper irreproducible. Resolution: immutable revisions, source-only audits, generated result macros, claim/evidence matrix, manifests, release inventory, and deterministic packaging are explicit deliverables. Remaining concern: gated data and hosted-model drift prevent full bitwise replay and are disclosed.

## Author-side audit

The paper can currently defend RQ1 because the evidence package is complete and hash-bound. RQ2, RQ4, and RQ5 cannot yet be defended as final scientific answers, regardless of operational progress. The manuscript's strongest residual editorial risk is breadth: if final results do not produce a coherent observability-ladder synthesis, the extensions should remain clearly separated rather than forced into one positive story.

