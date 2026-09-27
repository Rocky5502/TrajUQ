#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
BUDGET_HOURS="${BUDGET_HOURS:-72}"
MONITOR_SECONDS="${MONITOR_SECONDS:-3600}"
MAX_STAGE_ATTEMPTS="${MAX_STAGE_ATTEMPTS:-2}"
LOG_ROOT="${LOG_ROOT:-logs/a100_72h}"
STATE_PATH="${STATE_PATH:-results/a100_72h_autopilot_state.json}"
STAGE_FILE="${STAGE_FILE:-results/a100_72h_current_stage.txt}"

mkdir -p "$LOG_ROOT" results
START_EPOCH="$(date +%s)"
BUDGET_SECONDS="$((BUDGET_HOURS * 3600))"
DEADLINE_EPOCH="$((START_EPOCH + BUDGET_SECONDS))"
HEAD_SHA="$(git rev-parse HEAD)"
MONITOR_PID=""

write_state() {
  local status="$1"
  local stage="$2"
  local exit_code="${3:-0}"
  local now
  now="$(date +%s)"
  "$PYTHON" - "$STATE_PATH" "$status" "$stage" "$exit_code"     "$START_EPOCH" "$DEADLINE_EPOCH" "$now" "$HEAD_SHA" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output,
    status,
    stage,
    exit_code,
    start_epoch,
    deadline_epoch,
    now_epoch,
    head,
) = sys.argv[1:]
start_epoch = int(start_epoch)
deadline_epoch = int(deadline_epoch)
now_epoch = int(now_epoch)
payload = {
    "protocol": "trajuq_a100_72h_autopilot_v1",
    "status": status,
    "stage": stage,
    "exit_code": int(exit_code),
    "git_commit": head,
    "start_epoch": start_epoch,
    "deadline_epoch": deadline_epoch,
    "updated_epoch": now_epoch,
    "started_at_utc": datetime.fromtimestamp(
        start_epoch, tz=timezone.utc
    ).isoformat(),
    "deadline_at_utc": datetime.fromtimestamp(
        deadline_epoch, tz=timezone.utc
    ).isoformat(),
    "updated_at_utc": datetime.fromtimestamp(
        now_epoch, tz=timezone.utc
    ).isoformat(),
    "elapsed_hours": (now_epoch - start_epoch) / 3600.0,
    "remaining_hours": max(0.0, (deadline_epoch - now_epoch) / 3600.0),
}
path = Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

remaining_seconds() {
  local now
  now="$(date +%s)"
  local remaining="$((DEADLINE_EPOCH - now))"
  if (( remaining < 0 )); then
    remaining=0
  fi
  printf '%s\n' "$remaining"
}

monitor_loop() {
  local log="$LOG_ROOT/hourly_monitor.log"
  while true; do
    local now remaining stage
    now="$(date +%s)"
    remaining="$((DEADLINE_EPOCH - now))"
    if (( remaining < 0 )); then
      remaining=0
    fi
    stage="$(cat "$STAGE_FILE" 2>/dev/null || printf '%s' 'initializing')"
    {
      printf '\n===== %s =====\n' "$(date -Is)"
      printf 'stage=%s\n' "$stage"
      printf 'remaining_hours=%.2f\n' "$(awk "BEGIN {print $remaining/3600}")"
      printf 'git_head=%s\n' "$(git rev-parse HEAD 2>/dev/null || true)"
      printf '%s\n' '-- GPU --'
      nvidia-smi         --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw         --format=csv,noheader,nounits 2>&1 || true
      printf '%s\n' '-- disk --'
      df -h . 2>&1 || true
      printf '%s\n' '-- artifact sizes --'
      du -sh runs results data/processed 2>/dev/null || true
      printf '%s\n' '-- latest execution states --'
      for path in         results/primary_execution_state.json         results/primary_analysis_execution_state.json         results/adaptive_execution_state.json         results/repeated_sampling_execution_state.json
      do
        if [[ -f "$path" ]]; then
          printf '%s: ' "$path"
          "$PYTHON" - "$path" <<'PY' 2>/dev/null || true
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
x = json.loads(p.read_text(encoding="utf-8"))
print(
    "status=", x.get("status"),
    "complete_cells=", x.get("complete_cells"),
    "expected_cells=", x.get("expected_cells"),
)
PY
        fi
      done
    } >> "$log"
    if (( remaining <= 0 )); then
      break
    fi
    sleep "$MONITOR_SECONDS"
  done
}

cleanup() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

run_stage() {
  local stage="$1"
  local attempt=1
  printf '%s\n' "$stage" > "$STAGE_FILE"

  while (( attempt <= MAX_STAGE_ATTEMPTS )); do
    local remaining
    remaining="$(remaining_seconds)"
    if (( remaining <= 0 )); then
      write_state "BUDGET_EXHAUSTED" "$stage" 124
      return 124
    fi

    write_state "RUNNING" "$stage" 0
    local log="$LOG_ROOT/$(printf '%02d' "$attempt")_${stage}.log"
    printf '[TrajUQ 72h] stage=%s attempt=%d remaining=%.2fh log=%s\n'       "$stage" "$attempt" "$(awk "BEGIN {print $remaining/3600}")" "$log"

    set +e
    timeout --foreground --signal=TERM "${remaining}s"       bash scripts/a100_campaign_linux.sh "$stage"       > >(tee -a "$log") 2> >(tee -a "$log" >&2)
    local rc=$?
    set -e

    if (( rc == 0 )); then
      write_state "STAGE_PASS" "$stage" 0
      return 0
    fi
    if (( rc == 124 || rc == 137 || $(remaining_seconds) <= 0 )); then
      write_state "BUDGET_EXHAUSTED" "$stage" "$rc"
      return "$rc"
    fi

    write_state "RETRY_PENDING" "$stage" "$rc"
    printf '[TrajUQ 72h] stage=%s failed rc=%d; resumable retry %d/%d\n'       "$stage" "$rc" "$attempt" "$MAX_STAGE_ATTEMPTS" >&2
    attempt=$((attempt + 1))
    sleep 30
  done

  write_state "FAILED" "$stage" 1
  return 1
}

# Human-review gates must already be complete. The autopilot never fabricates
# or bypasses those attestations.
bash scripts/a100_campaign_linux.sh readiness

printf '%s\n' "initializing" > "$STAGE_FILE"
write_state "STARTING" "initializing" 0
monitor_loop &
MONITOR_PID="$!"

# Priority order preserves the core paper under a fixed rental envelope:
# full primary -> full analysis -> RQ3 -> expensive RQ2 -> paper rendering.
for stage in estimate primary analysis rq3 rq2 paper; do
  if ! run_stage "$stage"; then
    rc=$?
    if [[ "$(remaining_seconds)" -le 0 ]]; then
      printf '[TrajUQ 72h] rental budget exhausted during %s. Artifacts remain resumable.\n' "$stage" >&2
    else
      printf '[TrajUQ 72h] stopped after repeated failure in %s. Inspect the stage log; do not weaken scientific gates.\n' "$stage" >&2
    fi
    exit "$rc"
  fi
done

printf '%s\n' "complete" > "$STAGE_FILE"
write_state "PASS" "complete" 0
printf '[TrajUQ 72h] COMPLETE within %.2f elapsed hours.\n'   "$(awk "BEGIN {print ($(date +%s)-$START_EPOCH)/3600}")"
