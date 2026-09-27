#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
STAGE="${1:-status}"

fail() {
  printf '[TrajUQ] ERROR: %s\n' "$*" >&2
  exit 1
}

export_progress_handoff() {
  local label="$1"
  "$PYTHON" scripts/project_status.py
  "$PYTHON" scripts/export_research_handoff.py \
    --output "dist/trajuq_${label}_handoff.zip" \
    --skip-status-refresh
  printf '[TrajUQ] progress handoff: dist/trajuq_%s_handoff.zip\n' "$label"
}

require_clean_frozen_checkout() {
  "$PYTHON" scripts/validate_frozen_preprimary.py
  "$PYTHON" - <<'PY'
import json
import subprocess
from pathlib import Path

ledger_path = Path("results/revision_ledger.json")
gated_path = Path("results/gated_dataset_access.json")
contracts_path = Path("results/dataset_contracts.json")
if (
    not ledger_path.exists()
    or not gated_path.exists()
    or not contracts_path.exists()
):
    raise SystemExit(
        "Missing revision-ledger/gated-access/dataset-contract evidence; "
        "freeze pre-primary first."
    )

ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
gated = json.loads(gated_path.read_text(encoding="utf-8"))
contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
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

ledger_hash = __import__("hashlib").sha256(ledger_path.read_bytes()).hexdigest()
if (
    gated.get("protocol") != "trajuq_gated_dataset_access_v1"
    or gated.get("status") != "PASS"
    or gated.get("git_clean") is not True
    or gated.get("git_commit") != head
    or gated.get("revision_ledger_sha256") != ledger_hash
):
    raise SystemExit(
        "Gated dataset access is not a clean revision-bound PASS for this checkout."
    )

if (
    contracts.get("protocol") != "trajuq_dataset_contracts_v1"
    or contracts.get("status") != "PASS"
    or contracts.get("git_clean") is not True
    or contracts.get("git_commit") != head
    or contracts.get("revision_ledger_sha256") != ledger_hash
    or set(contracts.get("observed_datasets", []))
       != set(contracts.get("expected_datasets", []))
):
    raise SystemExit(
        "Dataset normalizer/scorer contracts are not a clean complete "
        "revision-bound PASS for this checkout."
    )

print(f"[TrajUQ] frozen checkout PASS: {head}")
PY
}

require_json_pass() {
  local path="$1"
  local protocol="$2"
  "$PYTHON" - "$path" "$protocol" <<'PY'
import json
import subprocess
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

head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True,
).strip()
artifact_commit = payload.get("git_commit")
if artifact_commit is not None and str(artifact_commit) != head:
    raise SystemExit(
        f"{path}: artifact commit {artifact_commit!r} != current HEAD {head!r}"
    )
if "git_clean" in payload and payload.get("git_clean") is not True:
    raise SystemExit(f"{path}: artifact was not recorded from a clean checkout")

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

prepare_primary_and_estimate() {
  require_primary_readiness
  prepare_missing_dataset_pairs
  "$PYTHON" scripts/plan_primary_matrix.py
  "$PYTHON" scripts/estimate_a100_campaign.py
}

run_primary() {
  prepare_primary_and_estimate
  "$PYTHON" scripts/execute_primary_plan.py
  "$PYTHON" scripts/execute_primary_plan.py --execute
  export_progress_handoff "primary"
}

run_analysis() {
  require_primary_readiness
  "$PYTHON" scripts/plan_primary_analysis.py
  "$PYTHON" scripts/execute_primary_analysis.py
  "$PYTHON" scripts/execute_primary_analysis.py --execute
  export_progress_handoff "analysis"
}

run_rq2() {
  require_primary_readiness
  "$PYTHON" scripts/plan_repeated_sampling.py
  "$PYTHON" scripts/execute_repeated_sampling.py
  "$PYTHON" scripts/execute_repeated_sampling.py --execute
  export_progress_handoff "rq2"
}

run_rq3() {
  require_primary_readiness
  "$PYTHON" scripts/plan_adaptive_runs.py
  "$PYTHON" scripts/execute_adaptive_plan.py
  "$PYTHON" scripts/execute_adaptive_plan.py --execute
  export_progress_handoff "rq3"
}

run_paper_render() {
  "$PYTHON" scripts/render_results_tex.py
  "$PYTHON" scripts/render_paper_result_figures.py
  "$PYTHON" scripts/audit_paper.py
  export_progress_handoff "paper"
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
    export_progress_handoff "pilot"
    printf '%s\n'       "[TrajUQ] STOP FOR HUMAN REVIEW."       "Inspect results/llada_gsm8k_pilot50_trace_review.md."       "Record the manual pilot attestation only after actual inspection."
    ;;

  parity-math)
    require_clean_frozen_checkout
    require_json_pass       "results/llada_gsm8k_pilot50_manual_review.json"       "trajuq_pilot_manual_gate_v2"
    bash scripts/parity_remaining_linux.sh all
    bash scripts/math500_gate_linux.sh prepare
    bash scripts/math500_gate_linux.sh pilot
    export_progress_handoff "parity_math"
    printf '%s\n'       "[TrajUQ] STOP FOR HUMAN MATH-500 REVIEW."       "Inspect all 25 records in results/math500_scorer_review.md."       "Record the scorer attestation only after actual inspection."
    ;;

  readiness)
    require_clean_frozen_checkout
    "$PYTHON" scripts/primary_readiness.py
    export_progress_handoff "readiness"
    ;;

  estimate)
    prepare_primary_and_estimate
    export_progress_handoff "estimate"
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
    fail "Unknown stage '$STAGE'. Use status|pilot|parity-math|readiness|estimate|primary|analysis|rq2|rq3|paper|all-after-readiness."
    ;;
esac
