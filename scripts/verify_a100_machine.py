#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


PROTOCOL = "trajuq_a100_machine_preflight_v1"
GIB = 1024 ** 3


def git_text(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def host_memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size
    except (AttributeError, OSError, ValueError):
        return None


def nvidia_smi() -> tuple[list[dict[str, str]], str | None]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"

    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 4:
            return [], f"unexpected nvidia-smi row: {line!r}"
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "driver_version": parts[2],
                "memory_total_mib": parts[3],
            }
        )
    return rows, None


def evaluate_machine(
    *,
    expected_gpus: int,
    required_name_substring: str,
    min_vram_gib: float,
    min_host_ram_gib: float,
    min_free_disk_gib: float,
) -> dict[str, Any]:
    failures: list[str] = []
    head = git_text("rev-parse", "HEAD")
    status = git_text("status", "--porcelain")
    clean = status == "" if status is not None else None

    if not head:
        failures.append("git HEAD could not be resolved")
    if clean is not True:
        failures.append("git worktree is not clean")

    torch_payload: dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "cuda_runtime": None,
        "device_count": 0,
        "device_name": None,
        "total_vram_bytes": None,
        "bf16_supported": None,
    }
    try:
        import torch

        torch_payload["installed"] = True
        torch_payload["version"] = str(torch.__version__)
        torch_payload["cuda_available"] = bool(torch.cuda.is_available())
        torch_payload["cuda_runtime"] = torch.version.cuda
        if not torch.cuda.is_available():
            failures.append("torch.cuda.is_available() is false")
        else:
            count = int(torch.cuda.device_count())
            torch_payload["device_count"] = count
            if count != expected_gpus:
                failures.append(
                    f"visible CUDA device count {count} != expected {expected_gpus}"
                )
            if count > 0:
                props = torch.cuda.get_device_properties(0)
                name = str(torch.cuda.get_device_name(0))
                total = int(props.total_memory)
                bf16 = bool(torch.cuda.is_bf16_supported())
                torch_payload["device_name"] = name
                torch_payload["total_vram_bytes"] = total
                torch_payload["bf16_supported"] = bf16
                if required_name_substring.lower() not in name.lower():
                    failures.append(
                        f"GPU name {name!r} does not contain "
                        f"{required_name_substring!r}"
                    )
                if total < int(min_vram_gib * GIB):
                    failures.append(
                        f"visible VRAM {total / GIB:.2f} GiB < "
                        f"required {min_vram_gib:.2f} GiB"
                    )
                if not bf16:
                    failures.append("torch reports bfloat16 is unsupported")
    except Exception as exc:
        failures.append(f"torch/CUDA probe failed: {type(exc).__name__}: {exc}")

    smi_rows, smi_error = nvidia_smi()
    if smi_error:
        failures.append(f"nvidia-smi probe failed: {smi_error}")
    elif len(smi_rows) != expected_gpus:
        failures.append(
            f"nvidia-smi reports {len(smi_rows)} GPUs != expected {expected_gpus}"
        )

    memory = host_memory_bytes()
    if memory is None:
        failures.append("host physical memory could not be resolved")
    elif memory < int(min_host_ram_gib * GIB):
        failures.append(
            f"host RAM {memory / GIB:.2f} GiB < required "
            f"{min_host_ram_gib:.2f} GiB"
        )

    disk = shutil.disk_usage(Path("."))
    if disk.free < int(min_free_disk_gib * GIB):
        failures.append(
            f"free disk {disk.free / GIB:.2f} GiB < required "
            f"{min_free_disk_gib:.2f} GiB"
        )

    return {
        "protocol": PROTOCOL,
        "status": "PASS" if not failures else "FAIL",
        "git_commit": head,
        "git_clean": clean,
        "requirements": {
            "expected_gpus": expected_gpus,
            "required_name_substring": required_name_substring,
            "min_vram_gib": min_vram_gib,
            "min_host_ram_gib": min_host_ram_gib,
            "min_free_disk_gib": min_free_disk_gib,
            "require_bf16": True,
        },
        "torch": torch_payload,
        "nvidia_smi": smi_rows,
        "host_ram_bytes": memory,
        "disk": {
            "total_bytes": int(disk.total),
            "free_bytes": int(disk.free),
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-fast machine suitability check for the frozen single-A100 campaign."
    )
    parser.add_argument("--expected-gpus", type=int, default=1)
    parser.add_argument("--required-name-substring", default="A100")
    parser.add_argument("--min-vram-gib", type=float, default=75.0)
    parser.add_argument("--min-host-ram-gib", type=float, default=40.0)
    parser.add_argument("--min-free-disk-gib", type=float, default=80.0)
    parser.add_argument(
        "--output",
        default="results/a100_machine_preflight.json",
    )
    args = parser.parse_args()

    if args.expected_gpus <= 0:
        raise SystemExit("--expected-gpus must be positive")
    for name, value in (
        ("--min-vram-gib", args.min_vram_gib),
        ("--min-host-ram-gib", args.min_host_ram_gib),
        ("--min-free-disk-gib", args.min_free_disk_gib),
    ):
        if value <= 0:
            raise SystemExit(f"{name} must be positive")

    payload = evaluate_machine(
        expected_gpus=args.expected_gpus,
        required_name_substring=args.required_name_substring,
        min_vram_gib=args.min_vram_gib,
        min_host_ram_gib=args.min_host_ram_gib,
        min_free_disk_gib=args.min_free_disk_gib,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(out)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
