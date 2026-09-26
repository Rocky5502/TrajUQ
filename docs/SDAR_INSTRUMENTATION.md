# SDAR instrumentation plan

Primary SDAR results are currently blocked until the pinned upstream runtime is instrumented and parity-tested.

Pinned upstream integration reference:

- repository: `JetAstra/SDAR`
- commit: `4c2749ba103448f45520e8411533710a1e66574d`
- selected checkpoint: `JetLM/SDAR-1.7B-Chat`

The public generator's core loop is `block_diffusion_generate`. TrajUQ telemetry should be attached without changing token-selection semantics.

## Hook points

For every denoising iteration in the active block:

1. obtain the upstream logits exactly as the normal generator does;
2. compute/log trajectory statistics **before** sampling/transfer;
3. run the unchanged upstream sampler;
4. apply the unchanged confidence/remasking transfer rule;
5. log the resulting block state if needed for commitment/revision bookkeeping.

The adapter must preserve:
- attention mask construction;
- position IDs;
- KV-cache behavior;
- number of transfer tokens;
- confidence threshold semantics;
- stopping criteria.

## Parity test

With telemetry disabled versus enabled, on the same checkpoint/config/seed:
- decoded output must match for deterministic settings;
- number of model forward calls must match;
- token-transfer decisions must match;
- timing differences may occur but generation semantics must not.

Only after this parity gate should SDAR move from `integration_boundary` to a runnable primary adapter.
