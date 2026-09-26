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

if [[ "$SKIP_INSTALL" != "1" ]]; then
  run_step "Install CPU/dev dependencies" "$PYTHON" -m pip install -e ".[dev,math]"
fi

run_step "Compile Python sources" "$PYTHON" -m compileall -q src scripts tests
run_step "Ruff" "$PYTHON" -m ruff check src tests scripts
run_step "Pytest" "$PYTHON" -m pytest -q
run_step "Metric smoke" "$PYTHON" scripts/smoke_metrics.py
run_step "End-to-end leak-safe smoke" "$PYTHON" scripts/smoke_end_to_end.py --output-dir results/smoke_e2e_local
run_step "MATH-500 scorer audit" "$PYTHON" scripts/audit_math500_scorer.py --output results/math500_scorer_audit.json
run_step "Paper anonymity audit" "$PYTHON" scripts/audit_paper.py

printf '\nLOCAL CPU VERIFICATION PASS\n'
printf 'This does not replace model-specific GPU parity gates.\n'
