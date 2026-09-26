#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an artifact-local model config with an immutable revision."
    )
    parser.add_argument("--template", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", default=None)
    args = parser.parse_args()

    revision = args.revision.strip()
    if not revision or revision.lower() == "null":
        raise SystemExit("Revision must be a non-empty immutable identifier.")

    template = Path(args.template)
    cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or not cfg.get("model_id"):
        raise SystemExit("Template must contain a model_id.")
    cfg["revision"] = revision

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    meta = {
        "protocol": "trajuq_frozen_model_config_v1",
        "template": str(template),
        "template_sha256": sha256(template),
        "output": str(out),
        "output_sha256": sha256(out),
        "model_id": cfg["model_id"],
        "adapter": cfg.get("adapter"),
        "revision": revision,
    }
    meta_out = (
        Path(args.metadata_output)
        if args.metadata_output
        else out.with_suffix(out.suffix + ".freeze.json")
    )
    meta_out.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
