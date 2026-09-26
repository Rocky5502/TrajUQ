#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except Exception:
        return None


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a machine-readable pre-run freeze manifest.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--dataset", action="append", default=[], help="Processed JSONL; may be repeated")
    parser.add_argument("--output", default="results/freeze_manifest.json")
    args = parser.parse_args()

    cfg_path = Path(args.model_config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    manifest = {
        "git_commit": git_head(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ["torch", "transformers", "datasets", "accelerate", "numpy", "scikit-learn"]
        },
        "model_config_path": str(cfg_path),
        "model_config_sha256": sha256(cfg_path),
        "model_config": cfg,
        "datasets": [{"path": str(Path(p)), "sha256": sha256(Path(p))} for p in args.dataset],
    }

    try:
        import torch

        manifest["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:
        manifest["cuda"] = {"available": False}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OK: freeze manifest -> {out}")


if __name__ == "__main__":
    main()
