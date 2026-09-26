#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--manifest", default="data/manifests/benchmarks.json")
    parser.add_argument("--cache-dir", default="data/cache")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit('Install extras first: pip install -e ".[models]"') from e

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    specs = manifest["datasets"]
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    for name in args.dataset:
        if name not in specs:
            raise SystemExit(f"Unknown dataset {name!r}. Available: {sorted(specs)}")
        spec = specs[name]
        print(
            f"[TrajUQ] downloading {name}: "
            f"{spec['hf_id']} / {spec.get('config')} / {spec['split']}"
        )
        if spec.get("gated"):
            print("[TrajUQ] gated dataset: accept upstream terms and authenticate locally.")

        if spec.get("config"):
            ds = load_dataset(spec["hf_id"], spec["config"], split=spec["split"])
        else:
            ds = load_dataset(spec["hf_id"], split=spec["split"])

        out = cache_root / name
        ds.save_to_disk(str(out))
        meta = {
            "dataset": name,
            "hf_id": spec["hf_id"],
            "config": spec.get("config"),
            "split": spec["split"],
            "fingerprint": getattr(ds, "_fingerprint", None),
            "num_rows": len(ds),
        }
        (out / "trajuq_download_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[TrajUQ] saved {len(ds)} rows to {out}")


if __name__ == "__main__":
    main()
