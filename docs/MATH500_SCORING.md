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
