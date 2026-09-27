#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
from importlib import metadata
from pathlib import Path

import yaml

from trajuq.io import read_jsonl


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


MIN_TRANSFORMERS_VERSION = (4, 52, 4)


def version_triplet(value: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\\d+)\\.(\\d+)\\.(\\d+)", value)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast preflight before a frozen GPU experiment.")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--allow-unpinned-revision", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    transformers_version: str | None = None
    try:
        transformers_version = metadata.version("transformers")
        observed = version_triplet(transformers_version)
        if observed is None or observed < MIN_TRANSFORMERS_VERSION:
            failures.append(
                "transformers runtime is too old: "
                f"{transformers_version!r}; require >=4.52.4 for the pinned SDAR runtime"
            )
    except metadata.PackageNotFoundError:
        failures.append("transformers is not installed; require >=4.52.4")

    cfg_path = Path(args.model_config)
    data_path = Path(args.data)
    if not cfg_path.exists():
        raise SystemExit(f"Missing model config: {cfg_path}")
    if not data_path.exists():
        raise SystemExit(f"Missing data file: {data_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    model_id = cfg.get("model_id")
    revision = cfg.get("revision")
    if not model_id:
        failures.append("model config has no model_id")
    if not revision and not args.allow_unpinned_revision:
        failures.append("model revision is not pinned")

    rows = list(read_jsonl(data_path))
    ids = [str(r.get("id", "")) for r in rows]
    if not rows:
        failures.append("data file is empty")
    if any(not x for x in ids):
        failures.append("one or more data rows have no id")
    if len(ids) != len(set(ids)):
        failures.append("duplicate data ids")
    if any("prompt" not in r for r in rows):
        failures.append("one or more data rows have no prompt")

    cuda_payload = {"required": bool(args.require_cuda), "available": None, "device": None}
    if args.require_cuda:
        try:
            import torch

            cuda_payload["available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                cuda_payload["device"] = torch.cuda.get_device_name(0)
            else:
                failures.append("CUDA required but torch.cuda.is_available() is false")
        except Exception as exc:
            failures.append(f"CUDA preflight failed: {exc}")

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "model_config": str(cfg_path),
        "model_config_sha256": sha256(cfg_path),
        "model_id": model_id,
        "model_revision": revision,
        "data": str(data_path),
        "data_sha256": sha256(data_path),
        "records": len(rows),
        "runtime": {
            "transformers": transformers_version,
            "minimum_transformers": "4.52.4",
        },
        "cuda": cuda_payload,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
