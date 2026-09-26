#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
STAGE="${1:-status}"

fail() {
  printf '[TrajUQ] ERROR: %s\n' "$*" >&2
  exit 1
}

require_clean_frozen_checkout() {
  "$PYTHON" - <<'PY'
import json
import subprocess
from pathlib import Path

ledger_path = Path("results/revision_ledger.json")
if not ledger_path.exists():
    raise SystemExit("Missing results/revision_ledger.json; freeze pre-primary first.")

ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
if ledger.get("protocol") != "trajuq_revision_ledger_v2":
    raise SystemExit("Revision ledger protocol is not v2.")
if ledger.get("status") != "PASS":
    raise SystemExit("Revision ledger is not PASS.")

head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True,
).strip()
status = subprocess.check_output(
    ["git", "status", "--porcelain"],
    text=True,
).strip()
if status:
    raise SystemExit(
        "Tracked worktree is dirty. Frozen experiment stages require a clean checkout."
    )
expected = str(ledger.get("git_commit", ""))
if head != expected:
    raise SystemExit(
        f"Current HEAD {head} differs from frozen experiment commit {expected}."
    )
print(f"[TrajUQ] frozen checkout PASS: {head}")
PY
}

require_json_pass() {
  local path="$1"
  local protocol="$2"
  "$PYTHON" - "$path" "$protocol" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
protocol = sys.argv[2]
if not path.exists():
    raise SystemExit(f"Missing required artifact: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("protocol") != protocol:
    raise SystemExit(
        f"{path}: protocol {payload.get('protocol')!r} != {protocol!r}"
    )
if payload.get("status") != "PASS":
    raise SystemExit(f"{path}: status is not PASS")
print(f"[TrajUQ] gate PASS: {path}")
PY
}

require_primary_readiness() {
  require_clean_frozen_checkout
  require_json_pass     "results/primary_readiness.json"     "trajuq_primary_readiness_v2"
}

prepare_missing_dataset_pairs() {
  "$PYTHON" - <<'PY'
import json
import subprocess
import sys
from pathlib import Path

ledger = json.loads(
    Path("results/revision_ledger.json").read_text(encoding="utf-8")
)
revisions = {
    str(item["name"]): str(item["resolved_sha"])
    for item in ledger["datasets"]
}
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
    present = [path.exists() for path in expected]
    if all(present):
        print(f"[PRESERVE] {name}: frozen pair already exists")
        continue
    if any(present):
        missing = [str(path) for path, ok in zip(expected, present) if not ok]
        raise SystemExit(
            f"{name}: partial frozen pair exists; missing={missing}. "
            "Inspect instead of overwriting."
        )
    revision = revisions.get(name)
    if not revision:
        raise SystemExit(f"{name}: no resolved revision in ledger")
    subprocess.check_call(
        [
            sys.executable,
            "scripts/prepare_eval_pair.py",
            "--dataset",
            name,
            "--revision",
            revision,
        ]
    )
PY
}

run_primary() {
  require_primary_readiness
  prepare_missing_dataset_pairs
  "$PYTHON" scripts/plan_primary_matrix.py
  "$PYTHON" scripts/execute_primary_plan.py
  "$PYTHON" scripts/execute_primary_plan.py --execute
}

run_analysis() {
  require_primary_readiness
  "$PYTHON" scripts/plan_primary_analysis.py
  "$PYTHON" scripts/execute_primary_analysis.py
  "$PYTHON" scripts/execute_primary_analysis.py --execute
}

run_rq2() {
  require_primary_readiness
  "$PYTHON" scripts/plan_repeated_sampling.py
  "$PYTHON" scripts/execute_repeated_sampling.py
  "$PYTHON" scripts/execute_repeated_sampling.py --execute
}

run_rq3() {
  require_primary_readiness
  "$PYTHON" scripts/plan_adaptive_runs.py
  "$PYTHON" scripts/execute_adaptive_plan.py
  "$PYTHON" scripts/execute_adaptive_plan.py --execute
}

run_paper_render() {
  "$PYTHON" scripts/render_results_tex.py
  "$PYTHON" scripts/render_paper_result_figures.py
  "$PYTHON" scripts/audit_paper.py
  "$PYTHON" scripts/project_status.py
  printf '%s\n'     "[TrajUQ] Result rendering complete."     "[TrajUQ] Commit the paper-side changes before submission_readiness.py."
}

case "$STAGE" in
  status)
    "$PYTHON" scripts/project_status.py
    ;;

  pilot)
    require_clean_frozen_checkout
    bash scripts/pilot_linux.sh prepare
    bash scripts/pilot_linux.sh parity
    bash scripts/pilot_linux.sh pilot
    printf '%s\n'       "[TrajUQ] STOP FOR HUMAN REVIEW."       "Inspect results/llada_gsm8k_pilot50_trace_review.md."       "Record the manual pilot attestation only after actual inspection."
    ;;

  parity-math)
    require_clean_frozen_checkout
    require_json_pass       "results/llada_gsm8k_pilot50_manual_review.json"       "trajuq_pilot_manual_gate_v2"
    bash scripts/parity_remaining_linux.sh all
    bash scripts/math500_gate_linux.sh prepare
    bash scripts/math500_gate_linux.sh pilot
    printf '%s\n'       "[TrajUQ] STOP FOR HUMAN MATH-500 REVIEW."       "Inspect all 25 records in results/math500_scorer_review.md."       "Record the scorer attestation only after actual inspection."
    ;;

  readiness)
    require_clean_frozen_checkout
    "$PYTHON" scripts/primary_readiness.py
    "$PYTHON" scripts/project_status.py
    ;;

  primary)
    run_primary
    ;;

  analysis)
    run_analysis
    ;;

  rq2)
    run_rq2
    ;;

  rq3)
    run_rq3
    ;;

  paper)
    run_paper_render
    ;;

  all-after-readiness)
    require_primary_readiness
    run_primary
    run_analysis
    run_rq2
    run_rq3
    run_paper_render
    ;;

  *)
    fail "Unknown stage '$STAGE'. Use status|pilot|parity-math|readiness|primary|analysis|rq2|rq3|paper|all-after-readiness."
    ;;
esac
