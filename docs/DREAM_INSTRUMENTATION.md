# Dream telemetry parity gate

Selected checkpoint: `Dream-org/Dream-v0-Instruct-7B`.

Dream exposes both `generation_logits_hook_func` and `generation_tokens_hook_func` through its public `diffusion_generate` interface. The upstream loop invokes the logits hook after computing the shifted predictive logits from the current masked state and **before token transfer**, whereas the token hook runs after transfer. TrajUQ therefore records telemetry only in `generation_logits_hook_func`, pairing each predictive distribution with the same pre-transfer unresolved state, and returns the logits unchanged.

This timing is required for the frozen decision-time baseline: a token committed on step (t) must retain the uncertainty from its last unresolved observation at step (t). Observing through the post-transfer token hook would incorrectly drop newly committed tokens from that decision state. Hook availability is still not enough: primary-paper use requires exact output parity to prove that telemetry does not alter generation.

## Frozen parity procedure

Use one exact model/tokenizer revision and the same 20 non-test prompts.

On Windows, the preferred guarded entry point is:

```powershell
PowerShell -ExecutionPolicy Bypass -File scripts/parity_remaining_windows.ps1 -Model dream
```

The wrapper requires the clean protocol-v1 CPU artifact and protocol-v2 revision ledger at the current frozen HEAD, prepares the frozen GSM8K calibration split if needed, preserves an already-valid parity-v3 PASS, and refuses to overwrite ambiguous partial evidence.


### 1. Native reference (no hook)

```bash
python scripts/run_dream_reference.py \
  --model Dream-org/Dream-v0-Instruct-7B \
  --revision <EXACT_MODEL_REVISION> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/dream_native_reference.jsonl \
  --limit 20 \
  --steps 128 \
  --max-new-tokens 128 \
  --temperature 0
```

### 2. Instrumented TrajUQ run

```bash
python scripts/run_generation.py \
  --adapter dream \
  --model Dream-org/Dream-v0-Instruct-7B \
  --revision <EXACT_MODEL_REVISION> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/dream_trajuq.raw.jsonl \
  --limit 20 \
  --steps 128 \
  --max-new-tokens 128
```

Audit the raw file:

```bash
python scripts/audit_run.py \
  --run runs/dream_trajuq.raw.jsonl \
  --stage raw \
  --require-revision
```

### 3. Exact parity comparison

```bash
python scripts/compare_run_parity.py \
  --reference runs/dream_native_reference.jsonl \
  --instrumented runs/dream_trajuq.raw.jsonl \
  --require-token-ids \
  --require-provenance \
  --output results/dream_parity.json
```

The hard gate is exact token-ID equality for all 20 prompts under the deterministic condition. The comparator emits `trajuq_exact_parity_v3`; primary readiness additionally requires exactly 20 shared records, uniform run provenance, the revision-ledger Dream SHA, and the same frozen experiment git commit.

## Telemetry inspection after parity

Dream has **no separate frozen 50-example pilot or human-attestation gate**.
The shared pre-primary human pilot gate is LLaDA + GSM8K; Dream's required
authorization is the exact parity-v3 PASS plus the normal raw-run semantic
audit at the frozen commit/revision.

After parity passes, it is still useful as a diagnostic to inspect several
Dream traces and confirm:
- observed trajectory count is compatible with requested denoising steps;
- logged generated-region length matches returned generated-token length;
- entropy/confidence/margin values are finite and non-constant;
- mask persistence decreases plausibly;
- late revisions are not an artifact of prompt-token inclusion;
- wall time and peak GPU memory are recorded.

This inspection must not be turned into post-hoc feature redesign or an
additional outcome-conditioned selection gate.


## Instrumentation overhead

Native and hook-instrumented runs record synchronized wall time and peak allocated GPU memory. After exact token parity passes:

```bash
python scripts/compare_instrumentation_overhead.py \
  --reference runs/dream_native_reference.jsonl \
  --instrumented runs/dream_trajuq.raw.jsonl \
  --output results/dream_instrumentation_overhead.json
```

Keep logger overhead separate from the underlying model's decoding latency.
