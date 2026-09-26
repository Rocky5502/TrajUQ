#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/environment.json")
    args = parser.parse_args()

    data = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_status_porcelain": git_value(["status", "--porcelain"]),
    }

    try:
        import numpy
        data["numpy"] = numpy.__version__
    except Exception:
        pass

    try:
        import torch
        data["torch"] = torch.__version__
        data["cuda_available"] = bool(torch.cuda.is_available())
        data["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            data["gpu_count"] = torch.cuda.device_count()
            data["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass

    try:
        import transformers
        data["transformers"] = transformers.__version__
    except Exception:
        pass

    try:
        import datasets
        data["datasets"] = datasets.__version__
    except Exception:
        pass

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
