#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
PYTHON="${PYTHON:-python}"

case "$STAGE" in
  prepare|pilot|all) ;;
  *)
    printf 'Usage: bash scripts/math500_gate_linux.sh [prepare|pilot|all]\n' >&2
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
    printf 'MATH-500 scorer-pilot artifacts already exist: %s\n' "${existing[*]}" >&2
    printf 'Inspect/archive them before a clean rerun; this wrapper will not overwrite evidence.\n' >&2
    exit 1
  fi
}

mkdir -p results runs

HEAD_SHA="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
  printf 'Worktree is dirty. Restore the frozen clean checkout before MATH-500 gating.\n' >&2
  exit 1
fi

run_step "Validate frozen pre-primary evidence" \
  "$PYTHON" scripts/validate_frozen_preprimary.py

"$PYTHON" - <<'PY'
import json
import subprocess
from pathlib import Path

head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
cpu_path=Path("results/cpu_verification.json")
ledger_path=Path("results/revision_ledger.json")
gated_path=Path("results/gated_dataset_access.json")
if not cpu_path.exists() or not ledger_path.exists() or not gated_path.exists():
    raise SystemExit("Run bash scripts/freeze_preprimary.sh first.")
cpu=json.loads(cpu_path.read_text(encoding="utf-8"))
ledger=json.loads(ledger_path.read_text(encoding="utf-8"))
gated=json.loads(gated_path.read_text(encoding="utf-8"))
failures=[]
if cpu.get("protocol")!="trajuq_cpu_verification_v1" or cpu.get("status")!="PASS":
    failures.append("CPU verification is not protocol-v1 PASS")
if cpu.get("git_clean") is not True or cpu.get("git_commit")!=head:
    failures.append("CPU verification is not clean/bound to HEAD")
if ledger.get("protocol")!="trajuq_revision_ledger_v2" or ledger.get("status")!="PASS":
    failures.append("revision ledger is not protocol-v2 PASS")
if ledger.get("git_commit")!=head:
    failures.append("revision ledger is not bound to HEAD")
if gated.get("protocol")!="trajuq_gated_dataset_access_v1" or gated.get("status")!="PASS":
    failures.append("gated dataset access is not protocol-v1 PASS")
if gated.get("git_clean") is not True or gated.get("git_commit")!=head:
    failures.append("gated dataset access is not clean/bound to HEAD")
if gated.get("revision_ledger_sha256")!=__import__("hashlib").sha256(ledger_path.read_bytes()).hexdigest():
    failures.append("gated dataset access is not bound to the current revision ledger")
if failures:
    raise SystemExit("Frozen-state validation failed:\n- "+"\n- ".join(failures))
PY

readarray -t VALUES < <("$PYTHON" - <<'PY'
import json
from pathlib import Path
ledger=json.loads(Path("results/revision_ledger.json").read_text(encoding="utf-8"))
models={str(x["adapter"]):x for x in ledger["models"]}
datasets={str(x["name"]):x for x in ledger["datasets"]}
for value in (
    ledger["git_commit"],
    models["llada"]["model_id"],
    models["llada"]["resolved_sha"],
    datasets["math500"]["resolved_sha"],
):
    print(value)
PY
)

FROZEN_COMMIT="${VALUES[0]}"
MODEL_ID="${VALUES[1]}"
MODEL_SHA="${VALUES[2]}"
DATASET_SHA="${VALUES[3]}"

CAL_DATA="data/processed/math500/calibration.jsonl"
TEST_DATA="data/processed/math500/test.jsonl"
SPLIT_MANIFEST="data/processed/math500/split_manifest.json"
SPLIT_AUDIT="data/processed/math500/split_audit.json"
SCORER_AUDIT="results/math500_scorer_audit.json"
RAW="runs/llada_math500_scorer_pilot.raw.jsonl"
SCORED="runs/llada_math500_scorer_pilot.scored.jsonl"
REVIEW="results/math500_scorer_review.md"
MANUAL="results/math500_scorer_manual_review.json"

"$PYTHON" - <<PY
import hashlib
import json
from pathlib import Path

p=Path("results/llada_parity.json")
if not p.exists():
    raise SystemExit("Missing results/llada_parity.json. LLaDA parity must PASS first.")
data=json.loads(p.read_text(encoding="utf-8"))
ref=Path(str(data.get("reference","")))
inst=Path(str(data.get("instrumented","")))
audit_path=inst.with_suffix(inst.suffix+".audit.json")

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

if not ref.exists() or not inst.exists() or not audit_path.exists():
    raise SystemExit("LLaDA parity evidence is incomplete.")
audit=json.loads(audit_path.read_text(encoding="utf-8"))
ok=(
    data.get("protocol")=="trajuq_exact_parity_v3"
    and data.get("status")=="PASS"
    and data.get("git_clean") is True
    and data.get("require_token_ids") is True
    and data.get("require_provenance") is True
    and data.get("git_commit")=="$FROZEN_COMMIT"
    and int(data.get("shared_records",-1))==20
    and int(data.get("mismatch_count",-1))==0
    and int(data.get("provenance_mismatch_count",-1))==0
    and data.get("run_provenance",{}).get("adapter")=="llada"
    and data.get("run_provenance",{}).get("model_revision")=="$MODEL_SHA"
    and data.get("reference_sha256")==sha(ref)
    and data.get("instrumented_sha256")==sha(inst)
    and audit.get("protocol")=="trajuq_run_audit_v1"
    and audit.get("status")=="PASS"
    and audit.get("git_clean") is True
    and audit.get("git_commit")=="$FROZEN_COMMIT"
    and audit.get("stage")=="raw"
    and audit.get("sha256")==sha(inst)
)
if not ok:
    raise SystemExit("LLaDA parity is not an exact hash/audit-bound PASS at frozen provenance.")
PY

require_scorer_audit() {
  if [[ ! -e "$SCORER_AUDIT" ]]; then
    run_step "MATH-500 frozen scorer audit"       "$PYTHON" scripts/audit_math500_scorer.py --output "$SCORER_AUDIT"
  fi
  "$PYTHON" - <<PY
import json
from pathlib import Path
p=Path("$SCORER_AUDIT")
a=json.loads(p.read_text(encoding="utf-8"))
deps=a.get("dependency_versions",{})
ok=(
    a.get("protocol")=="trajuq_math500_scorer_audit_v2"
    and a.get("status")=="PASS"
    and a.get("git_clean") is True
    and a.get("git_commit")=="$FROZEN_COMMIT"
    and int(a.get("failure_count",1))==0
    and deps.get("math-verify")=="0.9.0"
    and deps.get("antlr4-python3-runtime")=="4.13.2"
)
if not ok:
    raise SystemExit("MATH-500 automatic scorer audit is stale or not PASS.")
PY
}

assert_frozen_pair() {
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
        raise SystemExit(f"Missing frozen MATH-500 split artifact: {path}. Run prepare first.")
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
    raise SystemExit("Frozen MATH-500 split/audit is stale or invalid.")
PY
}

if [[ "$STAGE" == "prepare" || "$STAGE" == "all" ]]; then
  require_scorer_audit

  split_artifacts=("$CAL_DATA" "$TEST_DATA" "$SPLIT_MANIFEST" "$SPLIT_AUDIT")
  existing=0
  for path in "${split_artifacts[@]}"; do
    [[ -e "$path" ]] && existing=$((existing + 1))
  done
  if (( existing == 0 )); then
    run_step "Prepare frozen MATH-500 calibration/test pair"       "$PYTHON" scripts/prepare_eval_pair.py         --dataset math500         --revision "$DATASET_SHA"
  elif (( existing != ${#split_artifacts[@]} )); then
    printf 'Partial MATH-500 split artifacts exist. Inspect them before rerun.\n' >&2
    exit 1
  fi

  assert_frozen_pair
  run_step "MATH-500 LLaDA CUDA preflight"     "$PYTHON" scripts/preflight.py       --model-config results/frozen_models/llada.yaml       --data "$CAL_DATA"       --require-cuda
fi

if [[ "$STAGE" == "pilot" || "$STAGE" == "all" ]]; then
  require_scorer_audit
  assert_frozen_pair
  require_no_existing_artifacts     "$RAW" "$RAW.audit.json"     "$SCORED" "$SCORED.audit.json"     "$REVIEW" "$MANUAL"

  run_step "25-example LLaDA MATH-500 calibration pilot"     "$PYTHON" scripts/run_generation.py       --adapter llada       --model "$MODEL_ID"       --revision "$MODEL_SHA"       --input "$CAL_DATA"       --output "$RAW"       --limit 25       --seed 2027       --steps 128       --max-new-tokens 128       --block-length 128       --temperature 0       --llada-cfg-scale 0       --llada-remasking low_confidence

  run_step "Audit raw MATH-500 scorer pilot"     "$PYTHON" scripts/audit_run.py --run "$RAW" --stage raw --require-revision
  run_step "Score MATH-500 calibration pilot"     "$PYTHON" scripts/score_outputs.py       --run "$RAW" --input "$CAL_DATA" --output "$SCORED"
  run_step "Audit scored MATH-500 scorer pilot"     "$PYTHON" scripts/audit_run.py --run "$SCORED" --stage scored --require-revision
  run_step "Export deterministic 25-example MATH-500 review sheet"     "$PYTHON" scripts/export_math500_scorer_review.py       --scored "$SCORED"       --input "$CAL_DATA"       --count 25       --seed 2027       --output "$REVIEW"

  printf '\nAUTOMATED MATH-500 SCORER PILOT COMPLETE.\n'
  printf 'Manual gate remains: review %s, then run record_math500_scorer_review.py.\n' "$REVIEW"
fi

printf '\nFrozen commit: %s\n' "$FROZEN_COMMIT"
printf 'Automatic scorer audit: %s\n' "$SCORER_AUDIT"
