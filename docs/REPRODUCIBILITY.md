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
