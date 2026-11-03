# Preserved systems-diagnostic executors

These four files are the exact executed sources recovered from the rented A100
after all authorized systems diagnostics had completed. They are preserved
byte-for-byte; no algorithmic, measurement, path, or runtime behavior was
changed. The absolute `/data/...` paths are part of the executed environment,
contain no credential or host identifier, and are retained for provenance.

| File | SHA-256 |
|---|---|
| `TRAJUQ_GPU_SYSTEMS_BENCH_20261031.py` | `3b50a1b2a269cde3bfcbcae7c4cfe7011a0be2e1b7d903ec5bcb152940fa3faf` |
| `TRAJUQ_GPU_CONTEXT_DECODE_SCALING_20261031.py` | `473dd67b29dd48b7c9637549a98a23be0b9a65c4656b6da73ebbf800b2bd9698` |
| `TRAJUQ_GPU_ENDURANCE_20261031.py` | `b654f0098f671266153526be75bca8fb6af7e793e0c392958703cdc81f0967b6` |
| `TRAJUQ_WAIT_AND_PRESERVE_20261031.sh` | `5d3cad32c8909a28f6c3fed4a3e9fa6266d61d046240a80677894711d708a409` |

The resulting evidence is classified
`SECONDARY_EXPLORATORY_SYSTEMS_DIAGNOSTIC` with
`scientific_claim_authority=false`. These scripts must not be interpreted as
authorizing a new diagnostic or experiment.
