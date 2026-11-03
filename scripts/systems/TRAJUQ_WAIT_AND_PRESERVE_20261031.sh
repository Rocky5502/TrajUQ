#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/trajuq_systems_preservation_20261031"
mkdir -p "$ROOT"/{logs,hashes,archives,system}

echo "============================================================"
echo " TRAJUQ SYSTEMS DIAGNOSTICS PRESERVATION"
echo "============================================================"
date

echo
echo "Waiting for benchmark chain to finish..."

while pgrep -f 'TRAJUQ_GPU_SYSTEMS_BENCH_20261031.py|TRAJUQ_GPU_CONTEXT_DECODE_SCALING_20261031.py|TRAJUQ_GPU_ENDURANCE_20261031.py' >/dev/null; do
    date
    nvidia-smi \
      --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader
    sleep 60
done

echo
echo "ALL_GPU_DIAGNOSTICS_FINISHED"

echo
echo "===== COPY LOGS ====="

for f in \
  /data/TRAJUQ_GPU_SYSTEMS_BENCH_20261031.log \
  /data/TRAJUQ_GPU_CONTEXT_DECODE_SCALING_20261031.log \
  /data/TRAJUQ_GPU_ENDURANCE_20261031.log
do
    if [ -f "$f" ]; then
        cp -a "$f" "$ROOT/logs/"
    fi
done

echo
echo "===== SYSTEM SNAPSHOT ====="

{
    date
    uname -a
    echo
    cat /etc/os-release
    echo
    nvidia-smi
    echo
    /data/TrajUQ/.venv/bin/python --version
    /data/TrajUQ/.venv/bin/python - <<'PY'
import torch, transformers
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
} > "$ROOT/system/system_manifest.txt"

echo
echo "===== HASH DIAGNOSTIC OUTPUTS ====="

find \
  /data/trajuq_gpu_systems_bench_20261031 \
  /data/trajuq_gpu_context_decode_scaling_20261031 \
  /data/trajuq_gpu_endurance_20261031 \
  -type f \
  -print0 2>/dev/null \
| sort -z \
| xargs -0 sha256sum \
> "$ROOT/hashes/DIAGNOSTIC_FILES.sha256"

echo
echo "===== ARCHIVE ====="

tar -czf \
  "$ROOT/archives/trajuq_systems_diagnostics_20261031.tar.gz" \
  -C /data \
  trajuq_gpu_systems_bench_20261031 \
  trajuq_gpu_context_decode_scaling_20261031 \
  trajuq_gpu_endurance_20261031 \
  TRAJUQ_GPU_SYSTEMS_BENCH_20261031.log \
  TRAJUQ_GPU_CONTEXT_DECODE_SCALING_20261031.log \
  TRAJUQ_GPU_ENDURANCE_20261031.log

sha256sum \
  "$ROOT/archives/trajuq_systems_diagnostics_20261031.tar.gz" \
  > "$ROOT/archives/trajuq_systems_diagnostics_20261031.tar.gz.sha256"

echo
echo "===== FINAL INVENTORY ====="

du -sh \
  /data/trajuq_gpu_systems_bench_20261031 \
  /data/trajuq_gpu_context_decode_scaling_20261031 \
  /data/trajuq_gpu_endurance_20261031 \
  "$ROOT" 2>/dev/null || true

echo
sha256sum \
  "$ROOT/archives/trajuq_systems_diagnostics_20261031.tar.gz"

echo
echo "TRAJUQ_SYSTEMS_DIAGNOSTICS_PRESERVATION=PASS"
