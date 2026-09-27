# Reproducibility contract

Every reported number should be regenerable from a run manifest and immutable output JSONL.

## Required run metadata

- TrajUQ git commit
- upstream runtime commit
- model repository + exact revision
- dataset repository/config/split + revision/fingerprint
- prompt template version
- seed and full decoding configuration
- Python / NumPy / PyTorch / Transformers / CUDA versions
- GPU model/count
- start/end timestamps
- output SHA-256

Recommended path:

\`runs/<date>/<model>/<dataset>/<config>__seed-<seed>.jsonl\`

Do not overwrite completed runs. Resume only missing IDs and audit duplicates before evaluation.

## Determinism

Record Python/NumPy/Torch seeds, deterministic-algorithm setting, TF32 setting, dtype, and model/tokenizer revisions. For stochastic decoding, report repeated-seed sensitivity.

## Artifact policy

Commit code, configs, manifests, synthetic fixtures, and table/figure scripts.

Do not commit model weights, caches, gated benchmark rows, large raw trajectories, or credentials.


## Frozen pre-primary evidence

Before any GPU parity, pilot, or MATH-500 scorer run, validate the shared
pre-primary evidence set:

```bash
python scripts/validate_frozen_preprimary.py
```

A PASS requires, at one clean frozen experiment commit:

- clean CPU verification with hash-bound per-step logs;
- revision-ledger v2 with current primary-config/dataset-manifest hashes and
  exact frozen model-config path/SHA/revision/model identity;
- gated-dataset access v1 bound to the current revision-ledger SHA;
- dataset-contract v1 bound to the same ledger, covering all six primary
  datasets at their exact resolved revisions.

The dataset-contract gate streams one row from each frozen primary dataset,
applies the real normalizer, exercises the real frozen scorer using a
constructed correct final answer, and stores only schema/provenance/status. It
does not retain streamed benchmark question, answer, prompt, or reference
content.

All supported Windows/Linux parity, pilot, and MATH wrappers invoke this shared
validator before GPU work. A required code or evidence repair invalidates the
freeze; archive the old private evidence and create a new clean freeze rather
than overwriting it.
