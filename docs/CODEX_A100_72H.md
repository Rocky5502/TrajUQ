# Codex + single-A100 72-hour execution runbook

This runbook is for an autonomous coding/execution agent operating the frozen
TrajUQ experiment on one NVIDIA A100-SXM4-80GB for a maximum 72-hour rental
window.

## Non-negotiable rules

1. Never invent results or mark a gate PASS without its real artifact.
2. Never bypass the two human-review boundaries.
3. After the pre-primary freeze passes, do not edit tracked experiment code.
   Any required code repair means stop, preserve artifacts, commit the repair,
   start from a fresh results workspace, and freeze again.
4. Never tune features, thresholds, dataset membership, or reporting from final
   test outcomes.
5. Keep raw generation pre-outcome: references and correctness never enter raw
   trajectory records.
6. Never store Hugging Face tokens, SSH credentials, rental-provider
   credentials, or Overleaf credentials in the repository or logs.
7. Prefer shell-side monitoring over repeated agent polling. A healthy GPU run
   should continue even when Codex is disconnected.
8. Do not change `--device cuda` or `--dtype bfloat16` after freeze. They are
   part of the raw/repeated artifact contract, not merely launch preferences.

## Pre-rental repository state

Before copying the repository to the GPU machine:

- pull the latest `main`;
- run local CPU tests if practical;
- confirm the tracked dataset manifest contains the frozen
  `sha256_id_rank_v1` caps;
- confirm `configs/experiments/primary.yaml` declares the 72-hour budget;
- confirm the worktree is clean.

Do not prepare final benchmark JSONL on an arbitrary moving revision. The GPU
freeze resolves immutable model and dataset SHAs first; dataset preparation then
uses those exact ledger revisions.

## Server bootstrap

Use a persistent terminal session such as `tmux`.

```bash
git clone https://github.com/Rocky5502/TrajUQ.git
cd TrajUQ
git checkout <EXACT_COMMIT_SELECTED_FOR_FREEZE>
git status --porcelain

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,math,models,analysis]"

nvidia-smi
hf auth login
```

Authentication is interactive/operator-owned. Do not ask the user to paste
secrets into chat.

## Hard gates before autonomous scale

```bash
SKIP_INSTALL=1 bash scripts/freeze_preprimary.sh
bash scripts/a100_campaign_linux.sh pilot
```

Stop for the required 10-trace human review and record the attestation only
after the user actually inspects the review sheet.

Then:

```bash
bash scripts/a100_campaign_linux.sh parity-math
```

Stop for the required 25-example MATH-500 review and record the attestation only
after real inspection.

Then:

```bash
bash scripts/a100_campaign_linux.sh readiness
bash scripts/a100_campaign_linux.sh estimate
cat results/a100_compute_estimate.md
```

Readiness must be protocol-v2 PASS. If it fails, inspect the exact failed gate;
do not weaken it.

## 72-hour autonomous execution

Once readiness passes:

```bash
BUDGET_HOURS=72 MONITOR_SECONDS=3600 \
  bash scripts/a100_campaign_linux.sh autopilot-72h
```

The supervisor performs:

1. provenance-bound estimate;
2. complete capped 18-cell primary generation/scoring;
3. complete primary analysis and frozen statistics;
4. RQ3 adaptive-compute matrix;
5. five-seed repeated-sampling RQ2 comparator on the frozen 12-cell representative subset;
6. artifact-derived paper result rendering and paper audit.

This ordering protects the core paper before the expensive repeated-sampling
comparator. The repeated comparator is pre-registered on 12 cells: all three
primary models crossed with GSM8K, MATH-500, GPQA-Diamond, and MMLU-Pro.
ARC-Challenge and TruthfulQA remain mandatory in the full 18-cell primary and
RQ3 analyses. Large source roles are capped at 500 before model calls using the
frozen SHA-256 ID-ranking policy.

The supervisor:

- writes `results/a100_72h_autopilot_state.json`;
- writes the current stage to `results/a100_72h_current_stage.txt`;
- logs one GPU/disk/progress snapshot per hour to
  `logs/a100_72h/hourly_monitor.log`;
- maintains per-attempt stage logs under `logs/a100_72h/`;
- retries a resumable failed stage at most once by default;
- enforces the remaining wall-clock budget with `timeout`;
- leaves incomplete JSONL artifacts resumable rather than declaring success.

The RQ2 resident executor keeps each model resident across all frozen eligible
partitions, while the shared repeated runner preserves five separately audited
seed artifacts per partition and safely resumes contract-matching partial
prefixes.

## Hourly Codex policy

Codex does not need to stay active continuously. When an hourly check is
available, inspect only:

```bash
cat results/a100_72h_autopilot_state.json
tail -n 80 logs/a100_72h/hourly_monitor.log
nvidia-smi
df -h .
```

Use tokens only when one of these conditions occurs:

- autopilot state is `FAILED`, `RETRY_PENDING`, or `BUDGET_EXHAUSTED`;
- GPU utilization remains unexpectedly near zero while a GPU stage is marked
  RUNNING;
- disk free space is approaching an unsafe level;
- a process exited;
- a scientific audit/gate failed.

Do not repeatedly summarize healthy progress.

## Evidence preservation

After every major stage, the campaign wrapper exports a compact handoff ZIP
under `dist/`. The underlying `results/`, `runs/`, and
`data/processed/` remain authoritative and are intentionally gitignored.

Before releasing an ephemeral GPU instance, copy at least:

```text
results/
runs/
data/processed/
dist/
logs/a100_72h/
```

to persistent storage/local workstation.

## Paper and figures

Never type result numbers into `paper/main.tex` or
`paper/appendix.tex` manually. After the complete required result artifacts
exist, the campaign runs:

```bash
python scripts/render_results_tex.py
python scripts/render_paper_result_figures.py
python scripts/audit_paper.py
```

The renderer is the result-to-paper bridge. It must fail rather than fabricate
a missing result.

After paper-side generated files are reviewed and committed, run final
submission readiness when appropriate:

```bash
python scripts/submission_readiness.py
```

## Overleaf

The repository is the canonical manuscript source. Build the anonymous Overleaf
package from a clean checkout:

```bash
python scripts/build_overleaf_package.py
```

After all real results and figure manifests exist:

```bash
python scripts/build_overleaf_package.py --submission-ready
```

If the operator has independently configured an authenticated Overleaf Git
remote, Codex may sync the exact generated source commit through that remote.
Credentials must stay outside the repository. Otherwise upload the deterministic
ZIP. Never claim that live Overleaf was updated unless the remote push/upload
was actually verified.

## Completion definition

GPU work is complete only when the required artifacts report PASS/COMPLETE and
their source hashes/provenance validate. Paper completion additionally requires
artifact-derived tables/figures, the paper audit, and an exact anonymous
Overleaf package. The 72-hour target is an execution budget, not permission to
convert incomplete evidence into a positive result.
