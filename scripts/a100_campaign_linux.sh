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

require_pilot_manual_gate() {
  require_clean_frozen_checkout
  "$PYTHON" - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_ok(path: Path, stage: str, commit: str) -> bool:
    audit_path = path.with_suffix(path.suffix + ".audit.json")
    if not path.is_file() or not audit_path.is_file():
        return False
    try:
        audit = load(audit_path)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        audit.get("protocol") == "trajuq_run_audit_v1"
        and audit.get("status") == "PASS"
        and audit.get("git_clean") is True
        and audit.get("git_commit") == commit
        and audit.get("stage") == stage
        and str(audit.get("sha256", "")) == sha(path)
    )


manual_path = Path("results/llada_gsm8k_pilot50_manual_review.json")
if not manual_path.is_file():
    raise SystemExit(f"Missing required human pilot gate: {manual_path}")

manual = load(manual_path)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
failures: list[str] = []

if manual.get("protocol") != "trajuq_pilot_manual_gate_v2":
    failures.append("pilot manual gate protocol is not v2")
if manual.get("status") != "PASS":
    failures.append("pilot manual gate is not PASS")
if manual.get("git_clean") is not True or manual.get("git_commit") != head:
    failures.append("pilot manual gate is not clean/bound to current frozen HEAD")
if int(manual.get("reviewed_records", 0)) < 10:
    failures.append("pilot manual gate covers fewer than 10 reviewed records")
checks = manual.get("checks", {})
for key in (
    "trajectory_semantics",
    "scorer_audit",
    "parity_artifact_reviewed",
    "no_leakage_observed",
):
    if checks.get(key) is not True:
        failures.append(f"pilot manual gate check is not confirmed: {key}")

review = Path(str(manual.get("review_sheet", "")))
if (
    not review.is_file()
    or manual.get("review_sheet_sha256") != sha(review)
):
    failures.append("pilot review sheet is missing or hash-stale")

parity_path = Path(str(manual.get("parity_artifact", "")))
if not parity_path.is_file():
    failures.append("bound LLaDA parity artifact is missing")
else:
    parity = load(parity_path)
    if manual.get("parity_artifact_sha256") != sha(parity_path):
        failures.append("bound LLaDA parity artifact hash is stale")
    ref = Path(str(parity.get("reference", "")))
    inst = Path(str(parity.get("instrumented", "")))
    parity_ok = (
        parity.get("protocol") == "trajuq_exact_parity_v3"
        and parity.get("status") == "PASS"
        and parity.get("git_clean") is True
        and parity.get("git_commit") == head
        and parity.get("require_token_ids") is True
        and parity.get("require_provenance") is True
        and int(parity.get("shared_records", -1)) == 20
        and int(parity.get("mismatch_count", -1)) == 0
        and int(parity.get("provenance_mismatch_count", -1)) == 0
        and ref.is_file()
        and inst.is_file()
        and parity.get("reference_sha256") == (sha(ref) if ref.is_file() else None)
        and parity.get("instrumented_sha256") == (sha(inst) if inst.is_file() else None)
        and audit_ok(inst, "raw", head)
    )
    if not parity_ok:
        failures.append("bound LLaDA parity evidence is no longer a valid exact PASS")

quality_path = Path(str(manual.get("pilot_quality", "")))
if not quality_path.is_file():
    failures.append("bound pilot quality artifact is missing")
else:
    quality = load(quality_path)
    if manual.get("pilot_quality_sha256") != sha(quality_path):
        failures.append("bound pilot quality artifact hash is stale")
    raw = Path(str(quality.get("raw", "")))
    scored = Path(str(quality.get("scored", "")))
    quality_ok = (
        quality.get("protocol") == "trajuq_pilot_quality_gate_v2"
        and quality.get("status") == "PASS"
        and quality.get("git_clean") is True
        and quality.get("git_commit") == head
        and raw.is_file()
        and scored.is_file()
        and quality.get("raw_sha256") == (sha(raw) if raw.is_file() else None)
        and quality.get("scored_sha256") == (sha(scored) if scored.is_file() else None)
        and audit_ok(raw, "raw", head)
        and audit_ok(scored, "scored", head)
    )
    if not quality_ok:
        failures.append("bound pilot quality/raw/scored evidence is stale or invalid")
    if manual.get("review_scored_artifact") != str(scored):
        failures.append("manual review scored path differs from pilot quality source")
    if scored.is_file() and manual.get("review_scored_sha256") != sha(scored):
        failures.append("manual review scored SHA differs from current scored pilot")

if manual.get("pilot_raw_audit_ok") is not True:
    failures.append("manual gate did not attest a valid pilot raw audit")
if manual.get("pilot_scored_audit_ok") is not True:
    failures.append("manual gate did not attest a valid pilot scored audit")

if failures:
    raise SystemExit(
        "Pilot human-gate validation failed:\n- " + "\n- ".join(failures)
    )

print(f"[TrajUQ] source-bound human pilot gate PASS: {manual_path}")
PY
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
    require_pilot_manual_gate
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
