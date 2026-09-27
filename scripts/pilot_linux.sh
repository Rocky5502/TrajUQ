#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
PYTHON="${PYTHON:-python}"

case "$STAGE" in
  prepare|parity|pilot|all) ;;
  *)
    printf 'Usage: bash scripts/pilot_linux.sh [prepare|parity|pilot|all]\n' >&2
    exit 2
    ;;
esac

run_step() {
  local name="$1"
  shift
  printf '\n=== %s ===\n' "$name"
  "$@"
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_no_existing_artifacts() {
  local existing=()
  local path
  for path in "$@"; do
    if [[ -e "$path" ]]; then
      existing+=("$path")
    fi
  done
  if (( ${#existing[@]} > 0 )); then
    printf 'Existing partial/frozen artifacts require inspection before rerun: %s\n' "${existing[*]}" >&2
    printf 'This wrapper will not delete or overwrite evidence.\n' >&2
    exit 1
  fi
}

mkdir -p results runs

HEAD_SHA="$(git rev-parse HEAD)"
dirty="$(git status --porcelain)"
if [[ -n "$dirty" ]]; then
  printf 'Worktree is dirty. Restore the frozen clean checkout before GPU execution.\n' >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import json
import subprocess
from pathlib import Path

head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
cpu_path = Path("results/cpu_verification.json")
ledger_path = Path("results/revision_ledger.json")
if not cpu_path.exists() or not ledger_path.exists():
    raise SystemExit(
        "Missing frozen CPU/revision artifacts. Run bash scripts/freeze_preprimary.sh first."
    )

cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
failures = []
if cpu.get("protocol") != "trajuq_cpu_verification_v1":
    failures.append("CPU verification protocol is not v1")
if cpu.get("status") != "PASS" or cpu.get("git_clean") is not True:
    failures.append("CPU verification is not a clean PASS")
if cpu.get("git_commit") != head:
    failures.append("CPU verification commit differs from HEAD")
if ledger.get("protocol") != "trajuq_revision_ledger_v2":
    failures.append("revision ledger protocol is not v2")
if ledger.get("status") != "PASS":
    failures.append("revision ledger is not PASS")
if ledger.get("git_commit") != head:
    failures.append("revision ledger commit differs from HEAD")
if failures:
    raise SystemExit("Frozen-state validation failed:\n- " + "\n- ".join(failures))
PY

readarray -t FROZEN_VALUES < <("$PYTHON" - <<'PY'
import json
from pathlib import Path

ledger = json.loads(Path("results/revision_ledger.json").read_text(encoding="utf-8"))
models = {str(x["adapter"]): x for x in ledger["models"]}
datasets = {str(x["name"]): x for x in ledger["datasets"]}
print(ledger["git_commit"])
print(models["llada"]["model_id"])
print(models["llada"]["resolved_sha"])
print(datasets["gsm8k"]["resolved_sha"])
PY
)

FROZEN_COMMIT="${FROZEN_VALUES[0]}"
MODEL_ID="${FROZEN_VALUES[1]}"
MODEL_SHA="${FROZEN_VALUES[2]}"
DATASET_SHA="${FROZEN_VALUES[3]}"

MODEL_CONFIG="results/frozen_models/llada.yaml"
CAL_DATA="data/processed/gsm8k/calibration.jsonl"
TEST_DATA="data/processed/gsm8k/test.jsonl"
SPLIT_MANIFEST="data/processed/gsm8k/split_manifest.json"
SPLIT_AUDIT="data/processed/gsm8k/split_audit.json"
FREEZE_MANIFEST="results/pilot_freeze_manifest.json"
PARITY_REFERENCE="runs/llada_parity_reference.jsonl"
PARITY_TRAJUQ="runs/llada_parity_trajuq.raw.jsonl"
PARITY_RESULT="results/llada_parity.json"
PARITY_OVERHEAD="results/llada_instrumentation_overhead.json"
PILOT_RAW="runs/llada_gsm8k_pilot50.raw.jsonl"
PILOT_SCORED="runs/llada_gsm8k_pilot50.scored.jsonl"
PILOT_GATE="results/llada_gsm8k_pilot50_gate.json"
PILOT_REVIEW="results/llada_gsm8k_pilot50_trace_review.md"

printf 'FROZEN_GIT_COMMIT=%s\n' "$FROZEN_COMMIT"
printf 'MODEL_SHA=%s\n' "$MODEL_SHA"
printf 'DATASET_SHA=%s\n' "$DATASET_SHA"

assert_frozen_inputs() {
  "$PYTHON" - <<PY
import hashlib
import json
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

cal = Path("$CAL_DATA")
test = Path("$TEST_DATA")
split_path = Path("$SPLIT_MANIFEST")
audit_path = Path("$SPLIT_AUDIT")
freeze_path = Path("$FREEZE_MANIFEST")
model_cfg = Path("$MODEL_CONFIG")
for path in (cal, test, split_path, audit_path, freeze_path, model_cfg):
    if not path.exists():
        raise SystemExit(f"Missing frozen pilot input: {path}. Run prepare first.")

split = json.loads(split_path.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

failures = []
if split.get("requested_revision") != "$DATASET_SHA":
    failures.append("GSM8K revision differs from revision ledger")
if split.get("calibration_sha256") != sha(cal):
    failures.append("calibration hash differs from split manifest")
if split.get("test_sha256") != sha(test):
    failures.append("test hash differs from split manifest")
if (
    audit.get("protocol") != "trajuq_split_pair_audit_v1"
    or audit.get("status") != "PASS"
    or audit.get("requested_revision") != "$DATASET_SHA"
    or audit.get("calibration_sha256") != sha(cal)
    or audit.get("test_sha256") != sha(test)
    or int(audit.get("id_overlap_count", -1)) != 0
    or int(audit.get("exact_prompt_overlap_count", -1)) != 0
):
    failures.append("split audit is stale or not a zero-overlap PASS")
if freeze.get("git_commit") != "$FROZEN_COMMIT":
    failures.append("pilot freeze manifest belongs to another git commit")
if freeze.get("model_config_sha256") != sha(model_cfg):
    failures.append("pilot freeze manifest model-config hash drifted")

datasets = {str(x["path"]): str(x["sha256"]) for x in freeze.get("datasets", [])}
if datasets.get(str(cal)) != sha(cal):
    failures.append("pilot freeze manifest calibration binding drifted")
if datasets.get(str(test)) != sha(test):
    failures.append("pilot freeze manifest test binding drifted")

if failures:
    raise SystemExit("Frozen pilot input validation failed:\n- " + "\n- ".join(failures))
PY
}

parity_passes() {
  "$PYTHON" - <<PY
import hashlib
import json
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

path = Path("$PARITY_RESULT")
if not path.exists():
    raise SystemExit(1)
p = json.loads(path.read_text(encoding="utf-8"))
ref = Path(str(p.get("reference", "")))
inst = Path(str(p.get("instrumented", "")))
audit_path = inst.with_suffix(inst.suffix + ".audit.json")
if not ref.exists() or not inst.exists() or not audit_path.exists():
    raise SystemExit(1)
audit = json.loads(audit_path.read_text(encoding="utf-8"))
ok = (
    p.get("protocol") == "trajuq_exact_parity_v3"
    and p.get("status") == "PASS"
    and p.get("git_clean") is True
    and p.get("require_token_ids") is True
    and p.get("require_provenance") is True
    and p.get("git_commit") == "$FROZEN_COMMIT"
    and int(p.get("shared_records", -1)) == 20
    and int(p.get("mismatch_count", -1)) == 0
    and int(p.get("provenance_mismatch_count", -1)) == 0
    and p.get("run_provenance", {}).get("adapter") == "llada"
    and p.get("run_provenance", {}).get("model_revision") == "$MODEL_SHA"
    and p.get("reference_sha256") == sha(ref)
    and p.get("instrumented_sha256") == sha(inst)
    and audit.get("protocol") == "trajuq_run_audit_v1"
    and audit.get("status") == "PASS"
    and audit.get("git_clean") is True
    and audit.get("git_commit") == "$FROZEN_COMMIT"
    and audit.get("stage") == "raw"
    and audit.get("sha256") == sha(inst)
)
raise SystemExit(0 if ok else 1)
PY
}

pilot_passes() {
  "$PYTHON" - <<PY
import hashlib
import json
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

gate_path = Path("$PILOT_GATE")
raw = Path("$PILOT_RAW")
scored = Path("$PILOT_SCORED")
if not gate_path.exists() or not raw.exists() or not scored.exists():
    raise SystemExit(1)

gate = json.loads(gate_path.read_text(encoding="utf-8"))
raw_audit_path = raw.with_suffix(raw.suffix + ".audit.json")
scored_audit_path = scored.with_suffix(scored.suffix + ".audit.json")
if not raw_audit_path.exists() or not scored_audit_path.exists():
    raise SystemExit(1)
raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
scored_audit = json.loads(scored_audit_path.read_text(encoding="utf-8"))

def audit_ok(audit, artifact, stage):
    return (
        audit.get("protocol") == "trajuq_run_audit_v1"
        and audit.get("status") == "PASS"
        and audit.get("git_clean") is True
        and audit.get("git_commit") == "$FROZEN_COMMIT"
        and audit.get("stage") == stage
        and audit.get("sha256") == sha(artifact)
    )

ok = (
    gate.get("protocol") == "trajuq_pilot_quality_gate_v2"
    and gate.get("status") == "PASS"
    and gate.get("git_clean") is True
    and gate.get("git_commit") == "$FROZEN_COMMIT"
    and gate.get("raw") == str(raw)
    and gate.get("scored") == str(scored)
    and gate.get("raw_sha256") == sha(raw)
    and gate.get("scored_sha256") == sha(scored)
    and int(gate.get("raw_records", -1)) == 50
    and int(gate.get("scored_records", -1)) == 50
    and gate.get("adapters") == ["llada"]
    and gate.get("model_revisions") == ["$MODEL_SHA"]
    and audit_ok(raw_audit, raw, "raw")
    and audit_ok(scored_audit, scored, "scored")
)
raise SystemExit(0 if ok else 1)
PY
}

if [[ "$STAGE" == "prepare" || "$STAGE" == "all" ]]; then
  split_artifacts=("$CAL_DATA" "$TEST_DATA" "$SPLIT_MANIFEST" "$SPLIT_AUDIT")
  existing=0
  for path in "${split_artifacts[@]}"; do
    [[ -e "$path" ]] && existing=$((existing + 1))
  done

  if (( existing == 0 )); then
    run_step "Prepare frozen GSM8K pair"       "$PYTHON" scripts/prepare_eval_pair.py         --dataset gsm8k         --revision "$DATASET_SHA"
  elif (( existing != ${#split_artifacts[@]} )); then
    printf 'Partial GSM8K split artifacts exist. Inspect them before rerunning.\n' >&2
    exit 1
  fi

  run_step "CUDA preflight"     "$PYTHON" scripts/preflight.py       --model-config "$MODEL_CONFIG"       --data "$CAL_DATA"       --require-cuda

  if [[ ! -e results/pilot_environment.json ]]; then
    run_step "Environment snapshot"       "$PYTHON" scripts/environment_snapshot.py         --output results/pilot_environment.json
  fi

  if [[ ! -e "$FREEZE_MANIFEST" ]]; then
    run_step "Freeze pilot manifest"       "$PYTHON" scripts/freeze_manifest.py         --model-config "$MODEL_CONFIG"         --dataset "$CAL_DATA"         --dataset "$TEST_DATA"         --output "$FREEZE_MANIFEST"
  fi

  assert_frozen_inputs
fi

if [[ "$STAGE" == "parity" || "$STAGE" == "pilot" || "$STAGE" == "all" ]]; then
  assert_frozen_inputs
fi

if [[ "$STAGE" == "parity" || "$STAGE" == "all" ]]; then
  if parity_passes; then
    printf 'LLaDA parity already PASS and hash-valid; preserving evidence.\n'
    if [[ ! -e "$PARITY_OVERHEAD" ]]; then
      run_step "Instrumentation overhead"         "$PYTHON" scripts/compare_instrumentation_overhead.py           --reference "$PARITY_REFERENCE"           --instrumented "$PARITY_TRAJUQ"           --output "$PARITY_OVERHEAD"
    fi
  else
    require_no_existing_artifacts       "$PARITY_REFERENCE"       "$PARITY_TRAJUQ"       "$PARITY_TRAJUQ.audit.json"       "$PARITY_RESULT"       "$PARITY_OVERHEAD"

    run_step "20-prompt native LLaDA reference"       "$PYTHON" scripts/run_llada_reference.py         --model "$MODEL_ID"         --revision "$MODEL_SHA"         --input "$CAL_DATA"         --output "$PARITY_REFERENCE"         --limit 20         --steps 128         --max-new-tokens 128         --block-length 128         --temperature 0

    run_step "20-prompt TrajUQ telemetry"       "$PYTHON" scripts/run_generation.py         --adapter llada         --model "$MODEL_ID"         --revision "$MODEL_SHA"         --input "$CAL_DATA"         --output "$PARITY_TRAJUQ"         --limit 20         --steps 128         --max-new-tokens 128         --block-length 128         --temperature 0

    run_step "Audit LLaDA parity telemetry"       "$PYTHON" scripts/audit_run.py         --run "$PARITY_TRAJUQ"         --stage raw         --require-revision

    run_step "Exact LLaDA token parity v3"       "$PYTHON" scripts/compare_run_parity.py         --reference "$PARITY_REFERENCE"         --instrumented "$PARITY_TRAJUQ"         --require-token-ids         --require-provenance         --output "$PARITY_RESULT"

    if ! parity_passes; then
      printf 'LLaDA parity did not produce a reusable PASS artifact. STOP.\n' >&2
      exit 1
    fi

    run_step "Instrumentation overhead"       "$PYTHON" scripts/compare_instrumentation_overhead.py         --reference "$PARITY_REFERENCE"         --instrumented "$PARITY_TRAJUQ"         --output "$PARITY_OVERHEAD"
  fi
fi

if [[ "$STAGE" == "pilot" || "$STAGE" == "all" ]]; then
  if ! parity_passes; then
    printf 'Missing valid LLaDA parity PASS. Run parity first.\n' >&2
    exit 1
  fi

  if pilot_passes; then
    printf 'LLaDA + GSM8K 50-example pilot already PASS and hash-valid; preserving evidence.\n'
    if [[ ! -e "$PILOT_REVIEW" ]]; then
      run_step "Export 10-example manual review sheet"         "$PYTHON" scripts/export_trace_review.py           --scored "$PILOT_SCORED"           --output "$PILOT_REVIEW"           --count 10
    fi
  else
    require_no_existing_artifacts       "$PILOT_RAW"       "$PILOT_RAW.audit.json"       "$PILOT_SCORED"       "$PILOT_SCORED.audit.json"       "$PILOT_GATE"       "$PILOT_REVIEW"       results/llada_gsm8k_pilot50_manual_review.json

    run_step "50-example LLaDA + GSM8K pilot"       "$PYTHON" scripts/run_generation.py         --adapter llada         --model "$MODEL_ID"         --revision "$MODEL_SHA"         --input "$CAL_DATA"         --output "$PILOT_RAW"         --limit 50         --steps 128         --max-new-tokens 128         --block-length 128         --temperature 0

    run_step "Audit raw pilot"       "$PYTHON" scripts/audit_run.py         --run "$PILOT_RAW"         --stage raw         --require-revision

    run_step "Score pilot"       "$PYTHON" scripts/score_outputs.py         --run "$PILOT_RAW"         --input "$CAL_DATA"         --output "$PILOT_SCORED"

    run_step "Audit scored pilot"       "$PYTHON" scripts/audit_run.py         --run "$PILOT_SCORED"         --stage scored         --require-revision

    run_step "Automated pilot quality gate"       "$PYTHON" scripts/pilot_quality_gate.py         --raw "$PILOT_RAW"         --scored "$PILOT_SCORED"         --output "$PILOT_GATE"

    if ! pilot_passes; then
      printf 'Pilot quality gate did not produce a reusable PASS artifact. STOP.\n' >&2
      exit 1
    fi

    run_step "Export 10-example manual review sheet"       "$PYTHON" scripts/export_trace_review.py         --scored "$PILOT_SCORED"         --output "$PILOT_REVIEW"         --count 10
  fi

  printf '\nAUTOMATED LLaDA PILOT COMPLETE.\n'
  printf 'Manual gate remains: review %s, then run record_manual_gate.py.\n' "$PILOT_REVIEW"
fi
