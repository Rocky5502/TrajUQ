#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"

if [[ "$SKIP_INSTALL" != "1" ]]; then
  "$PYTHON" -m pip install -e ".[dev,math]"
fi

args=(
  scripts/verify_cpu.py
  --output results/cpu_verification.json
  --logs-dir results/cpu_verification_logs
)
if [[ "$ALLOW_DIRTY" == "1" ]]; then
  args+=(--allow-dirty)
fi

"$PYTHON" "${args[@]}"

printf '\nLOCAL CPU VERIFICATION PASS\n'
printf 'Artifact: results/cpu_verification.json\n'
printf 'This does not replace model-specific GPU parity gates.\n'
