# LLaDA-MoE telemetry parity gate

Primary checkpoint: `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`.

Reference sampler integration point:
- repository: `ML-GSAI/LLaDA`
- commit: `9182493720ed723ef8031210d85959364e51cbe0`
- sampler: public `generate.py` low-confidence masked-diffusion loop.

TrajUQ's LLaDA adapter mirrors the default public sampler and inserts telemetry only after model logits are produced and before the Gumbel/argmax proposal and token transfer.

## Hard parity protocol

Freeze one exact model/tokenizer revision and use 20 non-test prompts.

### 1. Native uninstrumented reference

```bash
python scripts/run_llada_reference.py \
  --model inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --revision <EXACT_MODEL_REVISION> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/llada_native_reference.jsonl \
  --limit 20 \
  --steps 128 \
  --max-new-tokens 128 \
  --block-length 128 \
  --temperature 0
```

### 2. Instrumented TrajUQ run

```bash
python scripts/run_generation.py \
  --adapter llada \
  --model inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --revision <EXACT_MODEL_REVISION> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/llada_trajuq.raw.jsonl \
  --limit 20 \
  --steps 128 \
  --max-new-tokens 128 \
  --block-length 128
```

### 3. Audit + exact comparison

```bash
python scripts/audit_run.py \
  --run runs/llada_trajuq.raw.jsonl \
  --stage raw \
  --require-revision

python scripts/compare_run_parity.py \
  --reference runs/llada_native_reference.jsonl \
  --instrumented runs/llada_trajuq.raw.jsonl \
  --require-token-ids \
  --require-provenance \
  --output results/llada_parity.json
```

The deterministic hard gate is exact token-ID equality on all 20 prompts. The comparator emits `trajuq_exact_parity_v3`; primary readiness additionally requires exactly 20 shared records, uniform run provenance, the revision-ledger model SHA, and the same frozen experiment git commit.

## Additional audit

- verify the tokenizer-derived mask ID for the MoE checkpoint;
- verify model/tokenizer revisions are identical between reference and instrumented runs;
- confirm logged trajectory length equals the intended total denoising steps;
- inspect entropy/confidence/revision dynamics on at least ten prompts;
- verify prompt positions are excluded from pooled trajectory features;
- record peak GPU memory and wall time;
- repeat a stochastic seed sensitivity check only after the deterministic gate passes.

No LLaDA-MoE result enters the paper before this gate passes.


## Instrumentation overhead

Both native and instrumented parity runners record synchronized per-example wall time and peak allocated GPU memory. After exact token parity passes, quantify telemetry overhead:

```bash
python scripts/compare_instrumentation_overhead.py \
  --reference runs/llada_native_reference.jsonl \
  --instrumented runs/llada_trajuq.raw.jsonl \
  --output results/llada_instrumentation_overhead.json
```

Report this overhead separately from model-generation throughput. A parity-safe logger can still be too expensive to justify at serving time.
