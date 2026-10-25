#!/usr/bin/env python
"""Capture read-only A100/environment/run evidence without persisting secrets."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
import sys

import paramiko


REMOTE_PROGRAM = r'''
import datetime, glob, hashlib, json, os, platform, subprocess

def sh(cmd):
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    return p.stdout.strip()

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def iso(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat()

def package_version(name):
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return None

raw=[]
for path in sorted(glob.glob('/data/TrajUQ/runs/**/*.raw.jsonl', recursive=True)):
    rows=0; seconds=0.0; model=None; dataset=None; peak=[]
    try:
        with open(path,encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                d=json.loads(line); rows += 1
                model = model or d.get('model')
                dataset = dataset or d.get('dataset')
                md=((d.get('generation') or {}).get('metadata') or {})
                v=md.get('wall_time_seconds')
                if isinstance(v,(int,float)): seconds += float(v)
                v=md.get('peak_gpu_memory_bytes') or md.get('max_memory_allocated_bytes')
                if isinstance(v,(int,float)): peak.append(int(v))
    except Exception as exc:
        raw.append({'path':os.path.relpath(path,'/data/TrajUQ'),'read_error':str(exc)})
        continue
    audit=path+'.audit.json'
    output_sha=sha(path)
    audit_payload={}
    if os.path.exists(audit):
        try:
            with open(audit,encoding='utf-8') as f: audit_payload=json.load(f)
        except Exception: audit_payload={}
    rel=os.path.relpath(path,'/data/TrajUQ')
    parts=rel.replace('\\','/').split('/')
    raw.append({
      'path':rel,'stage':parts[1] if len(parts)>1 else 'unknown',
      'model':model or (parts[2] if len(parts)>2 else None),
      'dataset':dataset or (parts[3] if len(parts)>3 else None),
      'split':parts[4] if len(parts)>4 else os.path.basename(path).split('.')[0], 'rows':rows,
      'generation_seconds':seconds, 'file_end_utc':iso(os.path.getmtime(path)),
      'output_sha256':output_sha, 'audit_present':os.path.exists(audit),
      'audit_sha256':sha(audit) if os.path.exists(audit) else None,
      'audit_status':audit_payload.get('status'),
      'audit_git_clean':audit_payload.get('git_clean'),
      'audit_git_commit':audit_payload.get('git_commit'),
      'audit_stage':audit_payload.get('stage'),
      'audit_binds_output':audit_payload.get('sha256') == output_sha,
      'peak_gpu_memory_bytes':max(peak) if peak else None,
    })

gpu_line=sh("nvidia-smi --query-gpu=name,memory.total,driver_version,utilization.gpu,power.draw --format=csv,noheader,nounits | head -1")
g=[x.strip() for x in gpu_line.split(',')]
cpu_model=sh("lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1")
mem=sh("awk '/MemTotal/{print $2*1024}' /proc/meminfo")
swap=sh("awk '/SwapTotal/{print $2*1024}' /proc/meminfo")
disk=sh("df -B1 --output=size,used,avail /data | tail -1")
import torch, transformers
out={
 'schema':'trajuq_current_environment_capture_v1',
 'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'host':'rented-a100-worker',
 'host_id_sha256':hashlib.sha256(platform.node().encode()).hexdigest(),
 'git':{'commit':sh('cd /data/TrajUQ && git rev-parse HEAD'),'dirty_count':int(sh("cd /data/TrajUQ && git status --porcelain | wc -l") or 0)},
 'os':{'description':sh(". /etc/os-release; printf '%s' \"$PRETTY_NAME\""),'kernel':platform.release(),'python':platform.python_version(),'glibc':platform.libc_ver()[1]},
 'runtime':{'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'transformers':transformers.__version__,'flash_attention':package_version('flash-attn'),'triton':package_version('triton')},
 'gpu':{'name':g[0] if g else None,'count':1,'memory_mib':int(g[1]) if len(g)>1 else None,'driver':g[2] if len(g)>2 else None,'utilization_percent_snapshot':float(g[3]) if len(g)>3 else None,'power_w_snapshot':float(g[4]) if len(g)>4 else None},
 'cpu':{'model':cpu_model,'logical_cpus':os.cpu_count(),'physical_cores':int(sh("lscpu | awk -F: '/Core\\(s\\) per socket/{gsub(/ /,\"\",$2);print $2}'") or 0)},
 'memory':{'ram_bytes':int(float(mem) if mem else 0),'swap_bytes':int(float(swap) if swap else 0)},
 'storage':dict(zip(('total_bytes','used_bytes','available_bytes'),map(int,disk.split()))) if disk else {},
 'active_processes':sh("ps -eo pid,lstart,etimes,args | grep -E 'resident_generate_plan|execute_primary_plan|primary_cpu_overlap' | grep -v grep").splitlines(),
 'raw_runs':raw,
}
print(json.dumps(out,sort_keys=True))
'''


def parse_credentials(path: Path) -> tuple[str, int, str, str]:
    text = path.read_text(encoding="utf-8")
    host_match = re.search(r"SSH[^\r\n\uff1a:]*[\uff1a:]\s*([0-9.]+)", text)
    port_match = re.search(r"(?:\u7aef\u53e3|port)[\uff1a:]\s*(\d+)", text, re.I)
    user_match = re.search(r"(?:SSH\u8d26\u6237|user(?:name)?)[\uff1a:]\s*([^\r\n]+)", text, re.I)
    password_match = re.search(r"(?:\u5bc6\u7801|password)[\uff1a:]\s*([^\r\n]+)", text, re.I)
    if not all((host_match, port_match, user_match, password_match)):
        raise ValueError("credential file does not match the supported provider format")
    return host_match.group(1), int(port_match.group(1)), user_match.group(1).strip(), password_match.group(1).strip()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--credentials", required=True, type=Path)
    p.add_argument("--output", default="results/compute_budget/current_environment.json", type=Path)
    args = p.parse_args()
    host, port, user, password = parse_credentials(args.credentials)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password, timeout=20)
    encoded = base64.b64encode(REMOTE_PROGRAM.encode()).decode()
    command = f"cd /data/TrajUQ && .venv/bin/python -c \"import base64;exec(base64.b64decode('{encoded}'))\""
    _, stdout, stderr = client.exec_command(command, timeout=180)
    payload = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    client.close()
    if error.strip():
        print(error, file=sys.stderr)
    data = json.loads(payload)
    atomic_json(args.output, data)
    print(json.dumps({"status":"PASS","captured_utc":data["captured_utc"],"raw_runs":len(data["raw_runs"])}, sort_keys=True))


if __name__ == "__main__":
    main()
