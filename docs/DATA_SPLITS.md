# Calibration/test split contract

TrajUQ distinguishes **calibration data** from the **final evaluation partition** before any model call.

Preferred official split pairs:

- GSM8K: calibration from `train`, final evaluation on `test`.
- ARC-Challenge: calibration from `validation`, final evaluation on `test`.
- MMLU-Pro: calibration from `validation`, final evaluation on `test`.

Benchmarks that expose only one usable public split for this study require a deterministic internal holdout:

- GPQA-Diamond: deterministic partition of its gated source split.
- TruthfulQA: deterministic partition of `validation`.
- MATH-500: deterministic partition of its 500-item `test` file **only for internal held-out analysis**. This is not the standard full MATH-500 benchmark score. MATH-500 is also blocked until the symbolic-equivalence scorer passes its audit.

Run:

```bash
python scripts/prepare_eval_pair.py \
  --dataset gsm8k \
  --revision <EXACT_DATASET_REVISION>
```

The script writes:

```text
data/processed/<dataset>/
  calibration.jsonl
  test.jsonl
  split_manifest.json
```

The split manifest records counts, requested revision, Hugging Face fingerprints, and SHA-256 hashes. Freeze it before generation.

Never obtain a final test result and then alter the calibration partition, feature set, prompt template, threshold rule, or scorer.
