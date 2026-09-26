# SDAR instrumentation and parity gate

Selected checkpoint: `JetLM/SDAR-1.7B-Chat`.

Pinned upstream integration reference:
- repository: `JetAstra/SDAR`
- commit: `4c2749ba103448f45520e8411533710a1e66574d`
- upstream generator: `block_diffusion_generate`

TrajUQ contains an instrumented research adapter derived from that MIT-licensed inference loop. Telemetry is collected from predictive logits **before token transfer**, so the logger never uses correctness or the future benchmark outcome.

## What must remain identical in the parity audit

- chat template and tokenization;
- mask token id;
- block-aligned attention mask;
- position IDs and prefill boundary;
- DynamicCache behavior;
- transfer-token counts;
- sampling/remasking rule;
- confidence threshold;
- stopping criteria;
- checkpoint/runtime revision;
- random seed.

## Native sampling semantics

The pinned upstream SDAR helper always samples with `torch.multinomial`. Its public default is `temperature=1.0`; passing zero would divide logits by zero and is **not** a valid native deterministic mode.

Accordingly, the hard parity gate uses the upstream-native stochastic path at `temperature=1.0` with the same seed in the reference and instrumented runs. TrajUQ telemetry performs no random draws, so exact token-ID equality is expected when instrumentation preserves sampler semantics.

TrajUQ also exposes `temperature=0` argmax as an explicit research control, but that control is not called upstream-native SDAR and is not used for the native parity claim.

## Exact blockwise feature aggregation

SDAR denoises one active block at a time. Each block exports mergeable sufficient statistics: unresolved-position entropy/margin/JS samples, per-step event/count statistics, per-token decision-time confidence, and commitment times.

TrajUQ then recomputes the example-level feature vector after concatenating blocks in generation order. It does **not** average block-level percentiles or temporal slopes. This preserves exact global p90/max/min features, late-stage rates, temporal trends, decision-time means, and commitment statistics under the same schema-v4 semantics used by whole-sequence adapters.

## Required parity experiment

Use one frozen checkpoint revision and 20 non-test prompts.

On Windows, the preferred guarded entry point is:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/parity_remaining_windows.ps1 -Model sdar
```

The wrapper requires the clean protocol-v1 CPU artifact and protocol-v2 revision ledger at the current frozen HEAD, prepares the frozen GSM8K calibration split if needed, preserves an already-valid parity-v3 PASS, and refuses to overwrite ambiguous partial evidence.


Reference:

```bash
python scripts/run_sdar_reference.py \
  --model JetLM/SDAR-1.7B-Chat \
  --revision <EXACT_MODEL_REVISION> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/sdar_native_reference.jsonl \
  --limit 20 \
  --max-new-tokens 128 \
  --block-length 4 \
  --denoising-steps 4 \
  --temperature 1.0 \
  --remasking-strategy low_confidence_dynamic
```

Instrumented run:

```bash
python scripts/run_generation.py \
  --adapter sdar \
  --model JetLM/SDAR-1.7B-Chat \
  --revision <EXACT_MODEL_REVISION> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/sdar_trajuq.raw.jsonl \
  --limit 20 \
  --max-new-tokens 128 \
  --block-length 4 \
  --sdar-denoising-steps 4 \
  --temperature 1.0 \
  --sdar-remasking-strategy low_confidence_dynamic
```

Audit and compare:

```bash
python scripts/audit_run.py \
  --run runs/sdar_trajuq.raw.jsonl \
  --stage raw \
  --require-revision

python scripts/compare_run_parity.py \
  --reference runs/sdar_native_reference.jsonl \
  --instrumented runs/sdar_trajuq.raw.jsonl \
  --require-token-ids \
  --require-provenance \
  --output results/sdar_parity.json

python scripts/compare_instrumentation_overhead.py \
  --reference runs/sdar_native_reference.jsonl \
  --instrumented runs/sdar_trajuq.raw.jsonl \
  --output results/sdar_instrumentation_overhead.json
```

The hard gate requires exact token equality for all 20 seed-matched prompts. The comparator emits `trajuq_exact_parity_v3`; primary readiness additionally requires uniform run provenance, the revision-ledger SDAR SHA, and the same frozen experiment git commit. Primary-paper SDAR results remain blocked until this passes.

## Pilot/primary configuration

After parity, use the pinned native temperature and other frozen model settings recorded in `configs/models/sdar.yaml` (or an artifact-local copy with the immutable model revision filled in). Do not silently switch between stochastic native sampling and the argmax research control after inspecting outcomes.
