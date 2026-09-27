#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-all}"
PYTHON="${PYTHON:-python}"

case "$MODEL" in
  sdar|dream|all) ;;
  *)
    printf 'Usage: bash scripts/parity_remaining_linux.sh [sdar|dream|all]\n' >&2
    exit 2
    ;;
esac

run_step() {
  local name="$1"
  shift
  printf '\n=== %s ===\n' "$name"
  "$@"
}

require_no_existing_artifacts() {
  local existing=()
  local path
  for path in "$@"; do
    [[ -e "$path" ]] && existing+=("$path")
  done
  if (( ${#existing[@]} > 0 )); then
    printf 'Existing parity artifacts require inspection before rerun: %s\n' "${existing[*]}" >&2
    printf 'This wrapper will not delete or overwrite evidence.\n' >&2
    exit 1
  fi
}

mkdir -p results runs

HEAD_SHA="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Worktree is dirty. Restore the frozen clean checkout before parity.\n' >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import json
import subprocess
from pathlib import Path

head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
cpu = json.loads(Path("results/cpu_verification.json").read_text(encoding="utf-8"))
ledger = json.loads(Path("results/revision_ledger.json").read_text(encoding="utf-8"))
failures = []
if cpu.get("protocol") != "trajuq_cpu_verification_v1" or cpu.get("status") != "PASS":
    failures.append("CPU verification is not protocol-v1 PASS")
if cpu.get("git_clean") is not True or cpu.get("git_commit") != head:
    failures.append("CPU verification is not clean/bound to HEAD")
if ledger.get("protocol") != "trajuq_revision_ledger_v2" or ledger.get("status") != "PASS":
    failures.append("revision ledger is not protocol-v2 PASS")
if ledger.get("git_commit") != head:
    failures.append("revision ledger is not bound to HEAD")
if failures:
    raise SystemExit("Frozen-state validation failed:\n- " + "\n- ".join(failures))
PY

readarray -t VALUES < <("$PYTHON" - <<'PY'
import json
from pathlib import Path
ledger=json.loads(Path("results/revision_ledger.json").read_text(encoding="utf-8"))
models={str(x["adapter"]):x for x in ledger["models"]}
datasets={str(x["name"]):x for x in ledger["datasets"]}
for value in (
    ledger["git_commit"],
    datasets["gsm8k"]["resolved_sha"],
    models["sdar"]["model_id"],
    models["sdar"]["resolved_sha"],
    models["dream"]["model_id"],
    models["dream"]["resolved_sha"],
):
    print(value)
PY
)

FROZEN_COMMIT="${VALUES[0]}"
DATASET_SHA="${VALUES[1]}"
SDAR_MODEL="${VALUES[2]}"
SDAR_SHA="${VALUES[3]}"
DREAM_MODEL="${VALUES[4]}"
DREAM_SHA="${VALUES[5]}"
CAL_DATA="data/processed/gsm8k/calibration.jsonl"
TEST_DATA="data/processed/gsm8k/test.jsonl"
SPLIT_MANIFEST="data/processed/gsm8k/split_manifest.json"
SPLIT_AUDIT="data/processed/gsm8k/split_audit.json"

"$PYTHON" - <<PY
import hashlib
import json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

cal=Path("$CAL_DATA")
test=Path("$TEST_DATA")
manifest=Path("$SPLIT_MANIFEST")
audit_path=Path("$SPLIT_AUDIT")
for path in (cal,test,manifest,audit_path):
    if not path.exists():
        raise SystemExit(f"Missing frozen GSM8K artifact: {path}. Run pilot_linux.sh prepare first.")
meta=json.loads(manifest.read_text(encoding="utf-8"))
audit=json.loads(audit_path.read_text(encoding="utf-8"))
ok=(
    meta.get("requested_revision")=="$DATASET_SHA"
    and meta.get("calibration_sha256")==sha(cal)
    and meta.get("test_sha256")==sha(test)
    and audit.get("protocol")=="trajuq_split_pair_audit_v1"
    and audit.get("status")=="PASS"
    and audit.get("requested_revision")=="$DATASET_SHA"
    and audit.get("calibration_sha256")==sha(cal)
    and audit.get("test_sha256")==sha(test)
    and int(audit.get("id_overlap_count",-1))==0
    and int(audit.get("exact_prompt_overlap_count",-1))==0
)
if not ok:
    raise SystemExit("Frozen GSM8K split/audit is stale or invalid.")
PY

parity_passes() {
  local adapter="$1"
  local revision="$2"
  local path="$3"
  "$PYTHON" - "$adapter" "$revision" "$path" "$FROZEN_COMMIT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

adapter, revision, path_text, commit = sys.argv[1:]
path=Path(path_text)
if not path.exists():
    raise SystemExit(1)
p=json.loads(path.read_text(encoding="utf-8"))
ref=Path(str(p.get("reference","")))
inst=Path(str(p.get("instrumented","")))
audit_path=inst.with_suffix(inst.suffix+".audit.json")
if not ref.exists() or not inst.exists() or not audit_path.exists():
    raise SystemExit(1)

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

audit=json.loads(audit_path.read_text(encoding="utf-8"))
ok=(
    p.get("protocol")=="trajuq_exact_parity_v3"
    and p.get("status")=="PASS"
    and p.get("git_clean") is True
    and p.get("require_token_ids") is True
    and p.get("require_provenance") is True
    and p.get("git_commit")==commit
    and int(p.get("shared_records",-1))==20
    and int(p.get("mismatch_count",-1))==0
    and int(p.get("provenance_mismatch_count",-1))==0
    and p.get("run_provenance",{}).get("adapter")==adapter
    and p.get("run_provenance",{}).get("model_revision")==revision
    and p.get("reference_sha256")==sha(ref)
    and p.get("instrumented_sha256")==sha(inst)
    and audit.get("protocol")=="trajuq_run_audit_v1"
    and audit.get("status")=="PASS"
    and audit.get("git_clean") is True
    and audit.get("git_commit")==commit
    and audit.get("stage")=="raw"
    and audit.get("sha256")==sha(inst)
)
raise SystemExit(0 if ok else 1)
PY
}

run_sdar() {
  local ref="runs/sdar_parity_reference.jsonl"
  local inst="runs/sdar_parity_trajuq.raw.jsonl"
  local parity="results/sdar_parity.json"
  local overhead="results/sdar_instrumentation_overhead.json"

  if parity_passes sdar "$SDAR_SHA" "$parity"; then
    printf 'SDAR parity already PASS and hash-valid; preserving evidence.\n'
  else
    require_no_existing_artifacts "$ref" "$inst" "$inst.audit.json" "$parity" "$overhead"
    run_step "SDAR 20-prompt native reference"       "$PYTHON" scripts/run_sdar_reference.py         --model "$SDAR_MODEL" --revision "$SDAR_SHA"         --input "$CAL_DATA" --output "$ref" --limit 20 --seed 2027         --max-new-tokens 128 --block-length 4 --denoising-steps 4         --temperature 1.0 --top-k 0 --top-p 1.0         --remasking-strategy low_confidence_dynamic --confidence-threshold 0.85
    run_step "SDAR 20-prompt TrajUQ telemetry"       "$PYTHON" scripts/run_generation.py         --adapter sdar --model "$SDAR_MODEL" --revision "$SDAR_SHA"         --input "$CAL_DATA" --output "$inst" --limit 20 --seed 2027         --max-new-tokens 128 --block-length 4 --sdar-denoising-steps 4         --temperature 1.0 --sdar-top-k 0 --top-p 1.0         --sdar-remasking-strategy low_confidence_dynamic         --sdar-confidence-threshold 0.85
    run_step "Audit SDAR parity telemetry"       "$PYTHON" scripts/audit_run.py --run "$inst" --stage raw --require-revision
    run_step "Exact SDAR token parity v3"       "$PYTHON" scripts/compare_run_parity.py         --reference "$ref" --instrumented "$inst"         --require-token-ids --require-provenance --output "$parity"
    if ! parity_passes sdar "$SDAR_SHA" "$parity"; then
      printf 'SDAR parity failed. STOP.\n' >&2
      exit 1
    fi
  fi
  if [[ ! -e "$overhead" ]]; then
    run_step "SDAR instrumentation overhead"       "$PYTHON" scripts/compare_instrumentation_overhead.py         --reference "$ref" --instrumented "$inst" --output "$overhead"
  fi
}

run_dream() {
  local ref="runs/dream_parity_reference.jsonl"
  local inst="runs/dream_parity_trajuq.raw.jsonl"
  local parity="results/dream_parity.json"
  local overhead="results/dream_instrumentation_overhead.json"

  if parity_passes dream "$DREAM_SHA" "$parity"; then
    printf 'Dream parity already PASS and hash-valid; preserving evidence.\n'
  else
    require_no_existing_artifacts "$ref" "$inst" "$inst.audit.json" "$overhead"
    run_step "Dream 20-prompt native reference"       "$PYTHON" scripts/run_dream_reference.py         --model "$DREAM_MODEL" --revision "$DREAM_SHA"         --input "$CAL_DATA" --output "$ref" --limit 20 --seed 2027         --steps 128 --max-new-tokens 128 --temperature 0 --top-p 1.0         --alg entropy --alg-temp 0.1
    run_step "Dream 20-prompt TrajUQ telemetry"       "$PYTHON" scripts/run_generation.py         --adapter dream --model "$DREAM_MODEL" --revision "$DREAM_SHA"         --input "$CAL_DATA" --output "$inst" --limit 20 --seed 2027         --steps 128 --max-new-tokens 128 --temperature 0 --top-p 1.0         --dream-alg entropy --dream-alg-temp 0.1
    run_step "Audit Dream parity telemetry"       "$PYTHON" scripts/audit_run.py --run "$inst" --stage raw --require-revision
    run_step "Exact Dream token parity v3"       "$PYTHON" scripts/compare_run_parity.py         --reference "$ref" --instrumented "$inst"         --require-token-ids --require-provenance --output "$parity"
    if ! parity_passes dream "$DREAM_SHA" "$parity"; then
      printf 'Dream parity failed. STOP.\n' >&2
      exit 1
    fi
  fi
  if [[ ! -e "$overhead" ]]; then
    run_step "Dream instrumentation overhead"       "$PYTHON" scripts/compare_instrumentation_overhead.py         --reference "$ref" --instrumented "$inst" --output "$overhead"
  fi
}

if [[ "$MODEL" == "sdar" || "$MODEL" == "all" ]]; then
  run_sdar
fi
if [[ "$MODEL" == "dream" || "$MODEL" == "all" ]]; then
  run_dream
fi

printf '\nREQUESTED REMAINING PARITY GATES PASS\n'
printf 'Frozen commit: %s\n' "$FROZEN_COMMIT"
printf 'Next: complete the manual pilot and MATH-500 gates, then run primary_readiness.py.\n'
