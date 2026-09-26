# GPU pilot gate

The first real TrajUQ experiment is deliberately small: **one checkpoint × one frozen calibration partition × 50 examples**.

## 1. Freeze before loading the model

1. Set an exact `revision` in the model YAML.
2. Download the benchmark with an exact dataset revision when possible.
3. Prepare the normalized JSONL.
4. Freeze calibration/test partitions before any model generation.
5. Run:

```bash
python scripts/preflight.py   --model-config configs/models/llada.yaml   --data data/processed/gsm8k_frozen/calibration.jsonl   --require-cuda
```

A primary run must not use `--allow-unpinned-revision`.

## 2. Run only 50 examples

Use a new output path. Never overwrite a completed run.

```bash
python scripts/run_generation.py   --adapter llada   --model inclusionAI/LLaDA-MoE-7B-A1B-Instruct   --revision <EXACT_REVISION>   --input data/processed/gsm8k_frozen/calibration.jsonl   --output runs/pilot_llada_gsm8k.raw.jsonl   --limit 50
```

## 3. Audit before scoring

```bash
python scripts/audit_run.py --run runs/pilot_llada_gsm8k.raw.jsonl --stage raw
```

Manually inspect at least ten rows. Check:
- decoded text is sensible;
- trajectory feature values are finite and non-constant;
- trajectory step count matches the intended sampler;
- model revision/mask ID are recorded;
- timing and peak GPU memory are present;
- no correctness/reference field leaked into the raw artifact.

## 4. Score into a new artifact

```bash
python scripts/score_outputs.py   --run runs/pilot_llada_gsm8k.raw.jsonl   --input data/processed/gsm8k_frozen/calibration.jsonl   --output runs/pilot_llada_gsm8k.scored.jsonl

python scripts/audit_run.py --run runs/pilot_llada_gsm8k.scored.jsonl --stage scored
```

## 5. Pilot decision

Scale only if all are true:
- raw and scored audits PASS;
- manual trace inspection finds no instrumentation mismatch;
- final entropy and trajectory risk have non-trivial variance;
- correctness parser is audited;
- rerunning feature extraction from the stored artifact is deterministic;
- no OOM/runtime instability remains.

If the LLaDA-MoE API is incompatible with the pilot adapter, stop and repair/parity-test the adapter. Do not silently substitute LLaDA-8B and continue as if it were the preregistered model.
