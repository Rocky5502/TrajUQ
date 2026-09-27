#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

run_step() {
  local name="$1"
  shift
  printf '\n=== %s ===\n' "$name"
  "$@"
}

mkdir -p results

dirty="$(git status --porcelain)"
if [[ -n "$dirty" ]]; then
  printf '%s\n' "Worktree is dirty. Commit or stash changes before freezing the experiment." >&2
  exit 1
fi

HEAD_SHA="$(git rev-parse HEAD)"
if [[ -z "$HEAD_SHA" ]]; then
  printf '%s\n' "Unable to resolve current git commit." >&2
  exit 1
fi
printf 'FROZEN_GIT_COMMIT=%s\n' "$HEAD_SHA"

existing_freeze_evidence=()
for path in \
  results/cpu_verification.json \
  results/revision_ledger.json \
  results/gated_dataset_access.json \
  results/dataset_contracts.json \
  results/math500_scorer_audit.json
do
  [[ -e "$path" ]] && existing_freeze_evidence+=("$path")
done
if [[ -d results/cpu_verification_logs ]]; then
  while IFS= read -r path; do
    existing_freeze_evidence+=("$path")
  done < <(find results/cpu_verification_logs -maxdepth 1 -type f -name '*.log' -print | sort)
fi
if [[ -d results/frozen_models ]]; then
  while IFS= read -r path; do
    existing_freeze_evidence+=("$path")
  done < <(find results/frozen_models -maxdepth 1 -type f -name '*.yaml' -print | sort)
fi
if (( ${#existing_freeze_evidence[@]} > 0 )); then
  printf '%s\n' "Refusing to start a mixed pre-primary freeze: existing immutable evidence was found." >&2
  printf '  %s\n' "${existing_freeze_evidence[@]}" >&2
  printf '%s\n' "Inspect/archive the existing evidence and use a fresh results target before freezing a new commit." >&2
  exit 1
fi

if [[ "$SKIP_INSTALL" != "1" ]]; then
  run_step "Install CPU/model/data verification dependencies" \
    "$PYTHON" -m pip install -e ".[dev,math,models]"
fi

run_step "A100 shell syntax verification" \
  bash -n \
    scripts/freeze_preprimary.sh \
    scripts/a100_campaign_linux.sh \
    scripts/a100_72h_autopilot.sh

run_step "Repeated runner standalone import smoke" \
  "$PYTHON" scripts/run_repeated_samples.py --help

run_step "Resident generator standalone import smoke" \
  "$PYTHON" scripts/resident_generate_plan.py --help

run_step "Clean-commit CPU verification" \
  "$PYTHON" scripts/verify_cpu.py \
    --output results/cpu_verification.json \
    --logs-dir results/cpu_verification_logs

run_step "Resolve immutable primary model/dataset revisions" \
  "$PYTHON" scripts/resolve_experiment_revisions.py \
    --output results/revision_ledger.json \
    --frozen-model-dir results/frozen_models

run_step "Verify frozen gated-dataset access" \
  "$PYTHON" scripts/verify_gated_dataset_access.py \
    --revision-ledger results/revision_ledger.json \
    --output results/gated_dataset_access.json

run_step "Verify exact-revision dataset normalizer/scorer contracts" \
  "$PYTHON" scripts/verify_dataset_contracts.py \
    --revision-ledger results/revision_ledger.json \
    --output results/dataset_contracts.json

HEAD_SHA="$HEAD_SHA" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

head = os.environ["HEAD_SHA"]
cpu = json.loads(Path("results/cpu_verification.json").read_text(encoding="utf-8"))
ledger = json.loads(Path("results/revision_ledger.json").read_text(encoding="utf-8"))
gated = json.loads(Path("results/gated_dataset_access.json").read_text(encoding="utf-8"))
contracts = json.loads(Path("results/dataset_contracts.json").read_text(encoding="utf-8"))

failures = []
if cpu.get("protocol") != "trajuq_cpu_verification_v1":
    failures.append(f"unexpected CPU protocol: {cpu.get('protocol')!r}")
if cpu.get("status") != "PASS":
    failures.append("CPU verification did not PASS")
if cpu.get("git_clean") is not True:
    failures.append("CPU verification was not recorded from a clean worktree")
if cpu.get("git_commit") != head:
    failures.append(
        f"CPU verification commit {cpu.get('git_commit')!r} != frozen HEAD {head!r}"
    )

if ledger.get("protocol") != "trajuq_revision_ledger_v2":
    failures.append(f"unexpected revision-ledger protocol: {ledger.get('protocol')!r}")
if ledger.get("status") != "PASS":
    failures.append("revision ledger did not PASS")
if ledger.get("git_commit") != head:
    failures.append(
        f"revision-ledger commit {ledger.get('git_commit')!r} != frozen HEAD {head!r}"
    )

if gated.get("protocol") != "trajuq_gated_dataset_access_v1":
    failures.append(f"unexpected gated-access protocol: {gated.get('protocol')!r}")
if gated.get("status") != "PASS":
    failures.append("gated dataset access did not PASS")
if gated.get("git_clean") is not True:
    failures.append("gated dataset access was not recorded from a clean worktree")
if gated.get("git_commit") != head:
    failures.append(
        f"gated-access commit {gated.get('git_commit')!r} != frozen HEAD {head!r}"
    )
if gated.get("revision_ledger_sha256") != __import__("hashlib").sha256(
    Path("results/revision_ledger.json").read_bytes()
).hexdigest():
    failures.append("gated dataset access is not bound to the current revision ledger")

if contracts.get("protocol") != "trajuq_dataset_contracts_v1":
    failures.append(f"unexpected dataset-contract protocol: {contracts.get('protocol')!r}")
if contracts.get("status") != "PASS":
    failures.append("dataset contracts did not PASS")
if contracts.get("git_clean") is not True:
    failures.append("dataset contracts were not recorded from a clean worktree")
if contracts.get("git_commit") != head:
    failures.append(
        f"dataset-contract commit {contracts.get('git_commit')!r} != frozen HEAD {head!r}"
    )
if contracts.get("revision_ledger_sha256") != __import__("hashlib").sha256(
    Path("results/revision_ledger.json").read_bytes()
).hexdigest():
    failures.append("dataset contracts are not bound to the current revision ledger")
if set(contracts.get("observed_datasets", [])) != set(contracts.get("expected_datasets", [])):
    failures.append("dataset-contract coverage is incomplete")

if failures:
    raise SystemExit("Pre-primary freeze validation failed:\n- " + "\n- ".join(failures))
PY

run_step "Validate complete frozen pre-primary evidence"   "$PYTHON" scripts/validate_frozen_preprimary.py

dirty_after="$(git status --porcelain)"
if [[ -n "$dirty_after" ]]; then
  printf '%s\n' "Tracked worktree changed during the pre-primary freeze. Do not trust the frozen artifacts." >&2
  exit 1
fi

run_step "Refresh project status" "$PYTHON" scripts/project_status.py

run_step "Export private pre-primary handoff" \
  "$PYTHON" scripts/export_research_handoff.py \
    --output dist/trajuq_preprimary_handoff.zip \
    --skip-status-refresh

printf '\nPRE-PRIMARY FREEZE PASS\n'
printf 'Frozen commit: %s\n' "$HEAD_SHA"
printf 'CPU artifact: results/cpu_verification.json\n'
printf 'Revision ledger: results/revision_ledger.json\n'
printf 'Gated dataset access: results/gated_dataset_access.json\n'
printf 'Dataset contracts: results/dataset_contracts.json\n'
printf 'Frozen model configs: results/frozen_models/\n'
printf 'Private handoff: dist/trajuq_preprimary_handoff.zip\n'
printf '\nNext: do not edit tracked experiment code. Run the first LLaDA parity gate from this same commit.\n'
