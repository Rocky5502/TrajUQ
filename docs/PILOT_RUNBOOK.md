# First GPU pilot runbook

The first pilot is frozen to the **LLaDA-MoE-7B-A1B-Instruct + GSM8K calibration** path. Do not scale to other models until the exact token-parity gate passes.

## A. Resolve immutable revisions

```bash
python scripts/resolve_hf_revision.py \
  --repo-type model \
  --repo-id inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --output results/llada_model_revision.json

python scripts/resolve_hf_revision.py \
  --repo-type dataset \
  --repo-id openai/gsm8k \
  --output results/gsm8k_dataset_revision.json
```

Copy the returned model SHA into `configs/models/llada.yaml: revision`. Pass the resolved dataset SHA to `prepare_eval_pair.py --revision`.

## B. Prepare the frozen calibration/test pair

```bash
python scripts/prepare_eval_pair.py \
  --dataset gsm8k \
  --revision <EXACT_GSM8K_SHA>
```

Do not regenerate these files after seeing model outcomes.

## C. Preflight and manifest

```bash
python scripts/preflight.py \
  --model-config configs/models/llada.yaml \
  --data data/processed/gsm8k/calibration.jsonl \
  --require-cuda

python scripts/environment_snapshot.py --output results/pilot_environment.json

python scripts/freeze_manifest.py \
  --model-config configs/models/llada.yaml \
  --dataset data/processed/gsm8k/calibration.jsonl \
  --dataset data/processed/gsm8k/test.jsonl \
  --output results/pilot_freeze_manifest.json
```

## D. Twenty-prompt hard parity gate

Run the native uninstrumented reference:

```bash
python scripts/run_llada_reference.py \
  --model inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --revision <EXACT_MODEL_SHA> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/llada_parity_reference.jsonl \
  --limit 20 \
  --steps 128 \
  --max-new-tokens 128 \
  --block-length 128 \
  --temperature 0
```

Run TrajUQ on the same prompts:

```bash
python scripts/run_generation.py \
  --adapter llada \
  --model inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --revision <EXACT_MODEL_SHA> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/llada_parity_trajuq.raw.jsonl \
  --limit 20 \
  --steps 128 \
  --max-new-tokens 128 \
  --block-length 128
```

Audit and compare:

```bash
python scripts/audit_run.py \
  --run runs/llada_parity_trajuq.raw.jsonl \
  --stage raw \
  --require-revision

python scripts/compare_run_parity.py \
  --reference runs/llada_parity_reference.jsonl \
  --instrumented runs/llada_parity_trajuq.raw.jsonl \
  --require-token-ids \
  --output results/llada_parity.json
```

**STOP if parity is not PASS.** Do not generate primary/pilot paper numbers until the adapter is repaired.

## E. Fifty-example pilot

After parity passes, use a fresh output file:

```bash
python scripts/run_generation.py \
  --adapter llada \
  --model inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --revision <EXACT_MODEL_SHA> \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/llada_gsm8k_pilot50.raw.jsonl \
  --limit 50 \
  --steps 128 \
  --max-new-tokens 128 \
  --block-length 128
```

Then:

```bash
python scripts/audit_run.py \
  --run runs/llada_gsm8k_pilot50.raw.jsonl \
  --stage raw \
  --require-revision

python scripts/score_outputs.py \
  --run runs/llada_gsm8k_pilot50.raw.jsonl \
  --input data/processed/gsm8k/calibration.jsonl \
  --output runs/llada_gsm8k_pilot50.scored.jsonl
```

Manually inspect at least ten trajectories before interpreting any UQ metric.

## Pilot pass criteria

Proceed to the full matrix only if:
- exact 20-prompt parity is PASS;
- raw audit is PASS;
- all features are finite;
- trajectory features show non-trivial variance;
- scorer output is manually plausible;
- no prompt/reference leakage is present;
- no repeated OOM/runtime instability exists;
- timing and peak GPU memory are present;
- the frozen code/data/model revisions are recorded.

A weak uncertainty signal is not a reason to redesign features using final test labels. It is a scientific result or a calibration-set design decision.
