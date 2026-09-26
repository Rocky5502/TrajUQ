# SDAR instrumentation and parity gate

Selected checkpoint: `JetLM/SDAR-1.7B-Chat`.

Pinned upstream integration reference:
- repository: `JetAstra/SDAR`
- commit: `4c2749ba103448f45520e8411533710a1e66574d`
- upstream generator: `block_diffusion_generate`

TrajUQ now contains `src/trajuq/adapters/sdar.py`, an instrumented research adapter derived from that MIT-licensed inference loop. Telemetry is collected from predictive logits **before token transfer** so the logger does not use outcome information.

## What must remain identical in the parity audit

- chat template and tokenization;
- mask token id;
- block-aligned attention mask;
- position IDs;
- prefill boundary and `DynamicCache` behavior;
- transfer-token counts;
- sampling/remasking rule;
- confidence threshold;
- stopping criteria;
- model/checkpoint revision.

## Telemetry semantics

SDAR denoises one active block at a time. TrajUQ computes the standard trajectory feature dictionary within each generated block and then token-weights those block summaries into an example-level record. `entropy_max` and `margin_min` preserve max/min semantics; `trajectory_steps` is summed.

This block-local definition is explicit because pretending SDAR exposes a whole-sequence denoising state would be scientifically incorrect.

## Required parity experiment

Use one frozen checkpoint revision and 20 non-test prompts.

1. Run the pinned upstream generator with a deterministic configuration.
2. Run TrajUQ with telemetry on and otherwise identical settings.
3. Compare:
   - decoded token IDs;
   - generated text;
   - number of model forward evaluations;
   - block transfer decisions (temporary debug log);
   - EOS stopping point.
4. Require exact token equality for deterministic greedy decoding.
5. For stochastic decoding, require equality under the same random seed after confirming both paths consume RNG identically; otherwise compare the deterministic condition for the hard parity gate.

Primary-paper SDAR results are blocked until this passes.

## Important difference from the upstream demo

The public upstream helper always samples with `torch.multinomial`. TrajUQ also supports `temperature=0` as a deterministic **research control** using argmax. That mode is not claimed to be byte-for-byte identical to an upstream sampling run. For upstream parity, use an equivalent deterministic upstream configuration or add the same argmax control to a temporary local upstream copy and document it.

## Pilot command

```bash
python scripts/run_generation.py \
  --adapter sdar \
  --model JetLM/SDAR-1.7B-Chat \
  --revision <EXACT_MODEL_REVISION> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/sdar_parity_pilot.raw.jsonl \
  --limit 20 \
  --max-new-tokens 128 \
  --block-length 4 \
  --sdar-denoising-steps 4 \
  --sdar-temperature 0
```

Then run `scripts/audit_run.py --stage raw --require-revision`.


## Automated parity comparison

Save the upstream/reference outputs and TrajUQ outputs as JSONL with matching `example_id` values and token IDs, then run:

```bash
python scripts/compare_run_parity.py \
  --reference runs/sdar_upstream_reference.jsonl \
  --instrumented runs/sdar_trajuq_instrumented.jsonl \
  --require-token-ids \
  --output results/sdar_parity.json
```

A hard parity gate requires `status=PASS`.
