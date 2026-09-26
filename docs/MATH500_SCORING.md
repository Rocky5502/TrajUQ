# MATH-500 scoring protocol

TrajUQ uses a **judge-free symbolic equivalence scorer**, not an LLM judge and not naive string matching.

Frozen dependency/configuration:
- package: `math-verify==0.9.0`
- extra: `antlr4_13_2`
- comparison precision: 6
- gold extraction: LaTeX + expression
- prediction extraction: LaTeX + expression
- boxed-answer priority: 0

Install:

```bash
pip install -e ".[math]"
```

## Normalization policy

For MATH-500:
1. the dataset's reference answer is wrapped in `\boxed{...}` unless already boxed;
2. if the model has an explicit `FINAL:` field, that field is wrapped in `\boxed{...}`;
3. otherwise the complete model output is passed to Math-Verify's extraction logic;
4. Math-Verify parses the expressions and performs symbolic/numeric equivalence;
5. parser/verification exceptions count as incorrect rather than being silently hand-fixed.

The wrapper is only an extraction aid; it does not change the mathematical expression.

## Why this scorer

The upstream Math-Verify project is designed for mathematical model evaluation and supports symbolic equivalence, numerical precision, sets/relations, matrices, and common LaTeX normalization. This avoids undercounting equivalent forms such as `1/2` vs `0.5` or `(x+1)^2` vs `x^2+2x+1`.

## Frozen regression cases

Repository tests cover:
- fractions;
- decimal/fraction equivalence;
- algebraically equivalent forms;
- boxed answers;
- radicals/numeric equivalence;
- an explicitly wrong fraction.

Before the primary MATH-500 run, execute the tests at the exact experiment commit and manually audit at least 25 examples spanning the benchmark's subject/level metadata.

## Known limitations

No automatic symbolic evaluator is perfect. Failure cases can include unusual malformed LaTeX, ambiguous answer strings, unsupported notation, or expressions that cause expensive symbolic simplification. Do not manually flip individual test labels after observing the result. If a systematic scorer defect is found, fix it using calibration/development examples, version the scoring protocol, and rerun the complete affected evaluation.

MATH-500 remains an **internal held-out evaluation** in TrajUQ's current split policy rather than a standard full MATH-500 leaderboard score.


## Reference-free repeated-sample clustering

The five-sample uncertainty baseline also covers MATH-500. It never consults the
benchmark reference when clustering sampled answers. For each example, all ten
pairs among the five generated answers are checked with the same pinned
Math-Verify equivalence relation. Exact normalized text equality is accepted
first; otherwise symbolic equivalence is evaluated. Connected components define
the vote classes used for disagreement, vote entropy, and unique-answer
fraction.

This is deliberately more conservative than naive string normalization and
keeps MATH-500 in the same 18-cell repeated-sampling matrix as the other tasks.

## Pre-primary scorer gates

On Windows, the preferred guarded path is:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/math500_gate_windows.ps1 -Stage all
```

This wrapper requires the clean CPU/revision freeze and an exact LLaDA parity-v3
PASS at the same frozen commit/revision. It verifies the installed
`math-verify==0.9.0` and `antlr4-python3-runtime==4.13.2`, prepares the frozen
MATH-500 calibration/test pair, runs exactly 25 calibration-only LLaDA examples,
audits raw/scored artifacts separately, and exports the deterministic review
sheet. It intentionally does **not** create the human attestation and refuses to
overwrite existing pilot/review evidence.

Before any primary MATH-500 generation, run the automatic audit at the same
frozen experiment commit recorded by revision-ledger v2:

```bash
python scripts/audit_math500_scorer.py
```

The resulting `trajuq_math500_scorer_audit_v2` artifact records the git commit
and must contain zero regression failures.

After LLaDA parity passes, generate and score a small MATH-500 **calibration-only**
pilot with at least 25 examples. Export a deterministic subject/level-diverse
review sheet:

```bash
python scripts/export_math500_scorer_review.py \
  --scored runs/llada_math500_scorer_pilot.scored.jsonl \
  --input data/processed/math500/calibration.jsonl \
  --count 25 \
  --output results/math500_scorer_review.md
```

Then manually inspect those calibration examples. Check extraction,
symbolic-equivalence behavior, and parser failures. Record that review with:

```bash
python scripts/record_math500_scorer_review.py \
  --review-sheet results/math500_scorer_review.md \
  --reviewed-records 25 \
  --answer-extraction-pass \
  --symbolic-equivalence-pass \
  --no-manual-label-flips \
  --notes "Summary of the frozen calibration-split scorer review."
```

The human recorder emits protocol
`trajuq_math500_manual_scorer_gate_v2`, binds the review sheet and automatic
audit by SHA-256, and requires the automatic audit to come from the same git
commit. `scripts/primary_readiness.py` re-verifies those hashes and blocks the
full primary matrix unless both artifacts are current PASSes. Do not inspect
final test labels to tune the scorer.
