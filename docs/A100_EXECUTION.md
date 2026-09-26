# NVIDIA A100 80GB execution path

This is the canonical Linux execution path for the frozen TrajUQ experiment.
It is intentionally strict: once the pre-primary freeze passes, do not edit
tracked experiment code, switch commits, regenerate frozen data after observing
model outcomes, or weaken a failed gate.

The target machine for this path is a Linux server with an NVIDIA A100 80GB.
Python >=3.10 is required; Python 3.11 is recommended when available.

## 0. Fresh checkout and Python environment

```bash
git clone https://github.com/Rocky5502/TrajUQ.git
cd TrajUQ
git status --porcelain
git rev-parse HEAD

nvidia-smi
python3.11 --version

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,math,models,analysis]"
```

Verify the runtime before spending GPU time:

```bash
python - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda=", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu=", torch.cuda.get_device_name(0))
    print("VRAM_GB=", round(p.total_memory / 1024**3, 2))
    print("bf16=", torch.cuda.is_bf16_supported())
PY
```

TrajUQ resolves immutable Hugging Face revisions during the freeze. Authenticate
locally before that step and never paste the token into chat, logs, or the
repository:

```bash
hf auth login
```

GPQA-Diamond is gated. The account used on the server must already have access
under the upstream dataset terms.

## 1. Freeze one immutable experiment commit

From a clean checkout:

```bash
SKIP_INSTALL=1 bash scripts/freeze_preprimary.sh
```

Require the final line `PRE-PRIMARY FREEZE PASS`. The freeze writes:

- `results/cpu_verification.json`;
- `results/revision_ledger.json`;
- `results/frozen_models/*.yaml`;
- `dist/trajuq_preprimary_handoff.zip`.

The CPU artifact and revision ledger must both be PASS and bind the exact same
git commit. From this point onward, do **not** edit tracked experiment code on
this run. If a code repair becomes necessary, invalidate the run, commit the
repair, freeze a new commit, and regenerate affected evidence.

## 2. LLaDA hard parity and 50-example GSM8K pilot

Run the guarded stages separately so a failed parity gate cannot be skipped:

```bash
bash scripts/pilot_linux.sh prepare
bash scripts/pilot_linux.sh parity
bash scripts/pilot_linux.sh pilot
```

The parity gate must be protocol `trajuq_exact_parity_v3` with exactly 20
shared prompts, zero output mismatches, and zero provenance mismatches. The
pilot then produces the 50-example raw/scored artifacts, audits, quality gate,
timing/VRAM evidence, and a deterministic ten-example review sheet.

Inspect:

```text
results/llada_gsm8k_pilot50_trace_review.md
```

Only after actually reviewing the ten records, record the human attestation:

```bash
python scripts/record_manual_gate.py \
  --review-sheet results/llada_gsm8k_pilot50_trace_review.md \
  --reviewed-records 10 \
  --trajectory-semantics-pass \
  --scorer-audit-pass \
  --parity-artifact-reviewed \
  --no-leakage-observed \
  --notes "Reviewed frozen 10-example pilot sheet on the A100 run."
```

Do not pass a flag that was not actually checked.

## 3. SDAR and Dream exact parity

Still on the same clean frozen checkout:

```bash
bash scripts/parity_remaining_linux.sh all
```

Both primary model families must pass exact parity-v3. A mismatch is a stop
condition; do not weaken the comparator or continue to the primary matrix.

## 4. MATH-500 scorer gate

Run the automatic scorer/dependency audit, frozen split preparation, CUDA
preflight, and 25-example calibration-only scoring pilot:

```bash
bash scripts/math500_gate_linux.sh prepare
bash scripts/math500_gate_linux.sh pilot
```

Inspect all 25 records in:

```text
results/math500_scorer_review.md
```

Then, and only then, record the manual scorer attestation:

```bash
python scripts/record_math500_scorer_review.py \
  --review-sheet results/math500_scorer_review.md \
  --reviewed-records 25 \
  --answer-extraction-pass \
  --symbolic-equivalence-pass \
  --no-manual-label-flips \
  --notes "Reviewed frozen 25-example MATH-500 scorer sheet."
```

The frozen dependency gate requires `math-verify==0.9.0` and
`antlr4-python3-runtime==4.13.2`.

## 5. Primary readiness hard gate

```bash
python scripts/primary_readiness.py
python scripts/project_status.py
```

Do not start the 18-cell primary matrix unless
`results/primary_readiness.json` uses protocol
`trajuq_primary_readiness_v2` and reports `status=PASS`.

## 6. Prepare only missing primary dataset pairs

GSM8K is already frozen by the pilot path and MATH-500 by its scorer gate. Do
not regenerate them after seeing outcomes. Prepare only dataset pairs that are
still absent, using the exact ledger revisions:

```bash
python - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

ledger = json.loads(Path("results/revision_ledger.json").read_text(encoding="utf-8"))
revisions = {str(x["name"]): str(x["resolved_sha"]) for x in ledger["datasets"]}
datasets = [
    "gsm8k",
    "math500",
    "gpqa_diamond",
    "arc_challenge",
    "mmlu_pro",
    "truthfulqa",
]

for name in datasets:
    root = Path("data/processed") / name
    expected = [
        root / "calibration.jsonl",
        root / "test.jsonl",
        root / "split_manifest.json",
        root / "split_audit.json",
    ]
    present = [p.exists() for p in expected]
    if all(present):
        print(f"[PRESERVE] {name}: frozen pair already exists")
        continue
    if any(present):
        raise SystemExit(f"Partial frozen pair exists for {name}; inspect it instead of overwriting.")
    subprocess.check_call([
        sys.executable,
        "scripts/prepare_eval_pair.py",
        "--dataset", name,
        "--revision", revisions[name],
    ])
PY
```

## 7. Freeze and execute the 18-cell primary matrix

```bash
python scripts/plan_primary_matrix.py
python scripts/execute_primary_plan.py
python scripts/execute_primary_plan.py --execute
```

The first executor call is a dry-run/status pass. The second resumes the exact
frozen plan. Existing evidence is reusable only when its protocol-v1 audit is
clean, stage-correct, commit-matched, and SHA-bound.

After the 18 calibration/test cells are COMPLETE:

```bash
python scripts/plan_primary_analysis.py
python scripts/execute_primary_analysis.py
python scripts/execute_primary_analysis.py --execute
```

This produces cell-level held-out metrics, nested calibrated comparison,
feature-family ablations, the training-free raw-risk diagnostic, the complete
equal-cell-weight macro result, and the pre-registered RQ1 claim gate. A
negative RQ1 gate is a valid scientific result and must not trigger post-hoc
analysis redesign.

## 8. RQ2 expensive five-sample comparator

Only after primary analysis predictions exist:

```bash
python scripts/plan_repeated_sampling.py
python scripts/execute_repeated_sampling.py
python scripts/execute_repeated_sampling.py --execute
```

This executes the frozen five-seed (2027--2031) comparator over all 18 cells.
Preserve compute accounting; do not report it as a free baseline.

## 9. RQ3 adaptive compute

After the refined primary artifacts and their audits are complete:

```bash
python scripts/plan_adaptive_runs.py
python scripts/execute_adaptive_plan.py
python scripts/execute_adaptive_plan.py --execute
```

The primary selection cap is 50%; 25% and 75% are frozen sensitivities. The
policy is selective second-pass rerun, not continuation from the base
trajectory.

## 10. Paper rendering and submission checks

Empirical values must enter the manuscript only through artifact-derived
rendering:

```bash
python scripts/render_results_tex.py
python scripts/render_paper_result_figures.py
python scripts/audit_paper.py --submission-ready
```

After committing or stashing the final paper-side changes so the submission
checkout is clean:

```bash
python scripts/submission_readiness.py
```

Do not type result numbers manually into `paper/main.tex`.

## Stop conditions

Stop rather than improvise if any of the following occurs:

- dirty worktree after the experiment freeze;
- CPU verification or revision ledger FAIL;
- model revision/data revision drift;
- any exact-parity mismatch;
- pilot quality or run-audit failure;
- missing/non-finite trajectory features;
- repeated OOM/runtime instability;
- MATH-500 scorer/manual-review failure;
- primary readiness not PASS;
- partial evidence that a guarded wrapper refuses to overwrite.

When a stop condition requires a code change, create a new commit and repeat the
freeze from the beginning. Never repair tracked experiment code underneath an
already-frozen run.
