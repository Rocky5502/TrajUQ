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


## ID namespacing

Prepared examples use IDs of the form:

`<dataset>:<immutable-source-split>:<upstream-item-id>`

This is required even for official train/validation/test pairs because many
datasets restart row numbering at zero in every split. Bare integer IDs would
create false calibration/test overlap during the leakage audit. Internal
holdouts retain their original source-split namespace on both sides; the
partition itself is still determined only by the frozen hash rule.


## Immutability and prompt versioning

Prepared split files are immutable by default. `prepare_eval_pair.py` refuses to
replace `calibration.jsonl`, `test.jsonl`, or `split_manifest.json` unless
`--overwrite` is explicitly supplied. Never use that override after model
outcomes have been inspected; create a new output root instead.

Primary preparation also requires an immutable dataset revision by default.
`--allow-unpinned-revision` exists only for exploratory work.

Every split manifest records `normalizer_version` in addition to content
SHA-256 hashes, dataset fingerprints, source split roles, and the requested
immutable revision. This makes prompt-format changes visible in provenance.
