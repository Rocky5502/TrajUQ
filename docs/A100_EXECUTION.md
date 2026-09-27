# NVIDIA A100 80GB execution path

This is the canonical Linux execution path for the frozen TrajUQ experiment.
It is intentionally strict: once the pre-primary freeze passes, do not edit
tracked experiment code, switch commits, regenerate frozen data after observing
model outcomes, or weaken a failed gate.

The target machine for this path is a Linux server with an NVIDIA A100 80GB.
Python >=3.10 is required; Python 3.11 is recommended when available. Always
replace `<EXACT_FREEZE_SHA_FROM_HANDOFF>` with the exact commit nominated in
the current project handoff; do not run the experiment from a moving branch.

## Rental profile and 72-hour strategy

The planned rental profile is one NVIDIA A100-SXM4-80GB GPU, 16 CPU cores,
48 GB host RAM, a 200 GB system disk, and approximately 250 Mbps network
bandwidth. The operational budget is **72 continuous hours**. This profile is
sufficient to load and execute each frozen primary checkpoint one at a time;
TrajUQ does not require multi-GPU model parallelism or training.

Generation remains single-GPU, but the A100 campaign uses resident checkpoint
execution to avoid needless reloads across frozen jobs.
Therefore one A100 is sufficient for the frozen study, and the 72-hour plan
reduces unnecessary calls through a predeclared dataset-cap policy rather than
changing outcomes after they are observed. The tracked manifest freezes
outcome-free SHA-256 ID ranking with seed 2027: GSM8K calibration/test are
capped at 500 each; ARC-Challenge and MMLU-Pro retain their smaller
calibration splits and cap the large test split at 500; the three smaller
single-split benchmarks keep their complete frozen holdouts.

Before scale, complete the pre-primary freeze, LLaDA parity/pilot, both human
review stops, SDAR/Dream parity, the MATH-500 gate, primary readiness, and then:

```bash
bash scripts/a100_campaign_linux.sh estimate
cat results/a100_compute_estimate.md
```

The estimator is the authoritative planning surface after real timing traces
exist. It reports the exact frozen generation-call count plus mean-timing and
P90-planning A100-hours and equivalent 24/7 single-A100 days. It excludes model
download/load startup, CPU analysis, failed retries, and optional RQ4. The
72-hour budget never authorizes weakening a scientific gate; if measured
throughput is slower than expected, the supervisor preserves resumable evidence
and reports the incomplete stage.

A 200 GB system disk is expected to be workable for the three checkpoints,
Python environment, prepared data, and experiment artifacts, but disk use is
not a scientific invariant. Check free space before model downloads and keep
the experiment evidence on persistent storage. If the provider offers a retained
data disk, use it; otherwise copy `results/`, `runs/`, `data/processed/`,
and each `dist/trajuq_*handoff.zip` off the instance before releasing it.
Never rely on an ephemeral system disk as the only copy of experiment evidence.

## 0. Fresh checkout and Python environment

```bash
git clone https://github.com/Rocky5502/TrajUQ.git
cd TrajUQ
git checkout <EXACT_FREEZE_SHA_FROM_HANDOFF>
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
under the upstream dataset terms. The pre-primary freeze now proves this on the
actual experiment machine by streaming one record at the exact ledger-resolved
revision. It stores only access/provenance metadata, never GPQA question or
answer content.

For long executions, run inside `tmux` or an equivalent persistent shell so a
network disconnect does not terminate the campaign. The guarded convenience
runner is:

```bash
bash scripts/a100_campaign_linux.sh status
```

It never bypasses human-review gates. The `pilot` stage stops for the manual
10-trace review; `parity-math` requires that pilot attestation and then stops
again for the 25-example MATH-500 review. Only after
`results/primary_readiness.json` is protocol-v2 PASS may scale execution
start. For the three-day rental, launch the autonomous supervisor:

```bash
BUDGET_HOURS=72 MONITOR_SECONDS=3600 \
  bash scripts/a100_campaign_linux.sh autopilot-72h
```

It prioritizes primary generation, primary analysis, and RQ3 before the
expensive repeated-sampling comparator, writes an hourly GPU/disk/progress log,
and does not require Codex or another agent to remain connected continuously.

Every successful campaign milestone now refreshes the machine-readable project
status and writes a privacy-scanned continuation bundle under `dist/`:
`trajuq_pilot_handoff.zip`, `trajuq_parity_math_handoff.zip`,
`trajuq_readiness_handoff.zip`, `trajuq_estimate_handoff.zip`,
`trajuq_primary_handoff.zip`, `trajuq_analysis_handoff.zip`,
`trajuq_rq2_handoff.zip`, `trajuq_rq3_handoff.zip`, and
`trajuq_paper_handoff.zip`. These bundles contain compact status/result
evidence and repository context, not raw run JSONL, processed benchmark rows,
model weights, or cache directories. They are continuity checkpoints only; they
do not replace the underlying frozen `results/*` evidence or make a failed
scientific gate pass.

## 1. Freeze one immutable experiment commit

From a clean checkout:

```bash
SKIP_INSTALL=1 bash scripts/freeze_preprimary.sh
```

Require the final line `PRE-PRIMARY FREEZE PASS`. Before writing anything,
the official freeze wrapper refuses to start if immutable evidence from another
freeze already exists (CPU artifact/logs, revision ledger/frozen model configs,
gated-access proof, six-dataset contract proof, or the automatic MATH-500 scorer audit). Archive the old
private handoff/evidence or use a fresh checkout/results workspace; do not mix
freeze commits.

The freeze writes:

- `results/cpu_verification.json`;
- `results/revision_ledger.json`;
- `results/gated_dataset_access.json`;
- `results/dataset_contracts.json`;
- `results/frozen_models/*.yaml`;
- `dist/trajuq_preprimary_handoff.zip`.

The CPU artifact, revision ledger, gated-dataset-access artifact, and live
dataset-contract artifact must all be PASS and bind the exact same git commit.
Both data-access artifacts must bind the exact revision-ledger SHA-256.
`dataset_contracts.json` streams four rows from each frozen primary dataset,
checks the normalizer/scorer contract on every sampled row, stores no benchmark
row content, and is rejected by the shared guard if it only contains legacy
one-row evidence. From this point onward, do **not** edit tracked experiment code on
this run. If a code repair becomes necessary, invalidate the run, archive the existing
private evidence, commit the repair, move to a fresh evidence workspace or
checkout, freeze the new commit, and regenerate the full affected evidence
chain. The official wrapper will not overwrite the old freeze.

## 2. LLaDA hard parity and 50-example GSM8K pilot

Run the guarded stages separately so a failed parity gate cannot be skipped:

```bash
bash scripts/a100_campaign_linux.sh pilot
```

This is equivalent to the three explicit wrapper stages:

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

After the pilot manual attestation is recorded, continue on the same clean
frozen checkout:

```bash
bash scripts/a100_campaign_linux.sh parity-math
```

Before launching those GPU stages, the campaign revalidates the pilot human
attestation against the current review-sheet SHA, pilot-quality SHA, raw/scored
pilot files and audits, LLaDA parity JSON, native/instrumented parity sources,
and instrumented audit at the frozen commit. A stale or hand-edited PASS
attestation is a stop condition.

This runs the remaining SDAR/Dream parity gates and the automated MATH-500
prepare/pilot work. The underlying parity-only command remains:

```bash
bash scripts/parity_remaining_linux.sh all
```

Both primary model families must pass exact parity-v3. A mismatch is a stop
condition; do not weaken the comparator or continue to the primary matrix.

## 4. MATH-500 scorer gate

If you used the convenience command
`bash scripts/a100_campaign_linux.sh parity-math` in Section 3, the automatic
MATH-500 prepare and 25-example pilot stages are already complete. **Do not run
the pilot a second time**: the guarded wrapper will correctly refuse to
overwrite the existing raw/scored/review evidence.

If instead you are running the stages manually and did not use
`parity-math`, run:

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
bash scripts/a100_campaign_linux.sh readiness
```

The explicit commands are still:

```bash
python scripts/primary_readiness.py
python scripts/project_status.py
```

Do not start the 18-cell primary matrix unless
`results/primary_readiness.json` uses protocol
`trajuq_primary_readiness_v2` and reports `status=PASS`.

## 6. Full post-readiness campaign

Once both human attestations exist and primary readiness is PASS, first prepare
the six frozen dataset pairs, freeze the 18-cell primary plan, and derive a
measured compute forecast without launching the matrix:

```bash
bash scripts/a100_campaign_linux.sh estimate
cat results/a100_compute_estimate.md
```

The estimate is bound to the clean experiment commit, exact prepared inputs, and
audited on-GPU traces: the 50-example LLaDA pilot plus 20-prompt SDAR/Dream
parity telemetry. It reports both a mean-timing projection and a conservative
P90-per-example planning sum. It is operational planning metadata, not a paper
result.

Then the complete remaining frozen campaign can be resumed stage-by-stage:

```bash
bash scripts/a100_campaign_linux.sh primary
bash scripts/a100_campaign_linux.sh analysis
bash scripts/a100_campaign_linux.sh rq2
bash scripts/a100_campaign_linux.sh rq3
bash scripts/a100_campaign_linux.sh paper
```

Or run the same sequence in one guarded invocation:

```bash
bash scripts/a100_campaign_linux.sh all-after-readiness
```

The runner re-checks the clean frozen checkout and readiness before GPU stages,
preserves already-complete audited artifacts, and aborts on any planner/executor
failure. It does not run the optional FLARE/RQ4 diagnostic. After the final
paper-render stage, commit the artifact-derived paper changes as a later clean
submission commit before running `scripts/submission_readiness.py`.

## 7. Prepare only missing primary dataset pairs

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

## 8. Freeze and execute the 18-cell primary matrix

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
feature-family ablations plus exact per-example group probabilities, the
training-free raw-risk diagnostic, prior-aligned trajectory-entropy and top-1
revision/flip diagnostics, the supplementary calibration-only isotonic
sensitivity with explicit ineligible-cell skips, the complete equal-cell-weight
primary macro, the complete 18-cell hierarchical feature-ablation macro, the
complete 18-cell prior-aligned signal diagnostic macro, the complete 18-cell
calibration-sensitivity PASS/SKIP macro, and the pre-registered RQ1 claim gate. A negative RQ1 gate is a valid scientific result
and must not trigger post-hoc analysis redesign.

## 9. RQ2 expensive five-sample comparator

Only after primary analysis predictions exist:

```bash
python scripts/plan_repeated_sampling.py
python scripts/execute_repeated_sampling.py
python scripts/resident_generate_repeated_plan.py \
  --plan results/repeated_sampling_plan.json
python scripts/execute_repeated_sampling.py --execute
```

This executes the frozen five-seed (2027--2031) comparator over the pre-registered 12-cell subset (all three models crossed with GSM8K, MATH-500, GPQA-Diamond, and MMLU-Pro). ARC-Challenge and TruthfulQA remain in the complete 18-cell primary/RQ3 analyses.
Repeated macro protocol v2 is fail-closed on compute accounting: every eligible cell must
retain valid test-time auxiliary wall-clock metadata, and the final artifact
reports the equal-cell-weight mean extra seconds per example with target
generation excluded. Do not report the comparator as a free baseline.

## 10. RQ3 adaptive compute

After the refined primary artifacts and their audits are complete:

```bash
python scripts/plan_adaptive_runs.py
python scripts/execute_adaptive_plan.py
python scripts/execute_adaptive_plan.py --execute
```

The primary selection cap is 50%; 25% and 75% are frozen sensitivities. The
policy is selective second-pass rerun, not continuation from the base
trajectory.

## 11. Paper rendering and submission checks

Empirical values must enter the manuscript only through artifact-derived
rendering. The text renderer v2 requires eight SHA-bound sources: the complete
primary aggregate, RQ1 gate, repeated-sampling aggregate, feature-ablation
aggregate, prior-aligned signal aggregate, supplementary isotonic calibration-
sensitivity aggregate, primary 50% adaptive aggregate, and frozen primary
config. The isotonic source populates an appendix-only sensitivity table and
does not alter the frozen RQ1 claim gate. The figure renderer independently binds the primary
aggregate/config plus all three adaptive-cap artifacts before replacing any TBD
result surface:

```bash
python scripts/render_results_tex.py
python scripts/render_paper_result_figures.py
python scripts/audit_paper.py --submission-ready
```

After committing the artifact-derived final paper-side changes so the
submission checkout is clean:

```bash
python scripts/submission_readiness.py
```

Do not stash those final rendered changes away before the gate; the readiness
check must see the exact rendered manuscript and figure manifests that will be
packaged.

Do not type result numbers manually into `paper/main.tex`.

## Stop conditions

Stop rather than improvise if any of the following occurs:

- dirty worktree after the experiment freeze;
- CPU verification, revision ledger, exact-revision gated-dataset access, or six-dataset normalizer/scorer contract FAIL;
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
