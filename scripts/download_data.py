#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_revision_overrides(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("--revision must use dataset=revision, e.g. gsm8k=<commit>")
        name, revision = value.split("=", 1)
        name = name.strip()
        revision = revision.strip()
        if not name or not revision:
            raise SystemExit(f"Invalid --revision value: {value!r}")
        out[name] = revision
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument("--manifest", default="data/manifests/benchmarks.json")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        help="Optional dataset=revision override; may be repeated.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit('Install extras first: pip install -e ".[models]"') from e

    overrides = parse_revision_overrides(args.revision)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    specs = manifest["datasets"]
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    for name in args.dataset:
        if name not in specs:
            raise SystemExit(f"Unknown dataset {name!r}. Available: {sorted(specs)}")
        spec = specs[name]
        revision = overrides.get(name, spec.get("revision"))
        print(
            f"[TrajUQ] downloading {name}: "
            f"{spec['hf_id']} / {spec.get('config')} / {spec['split']} / revision={revision}"
        )
        if spec.get("gated"):
            print("[TrajUQ] gated dataset: accept upstream terms and authenticate locally.")

        kwargs = {"split": spec["split"]}
        if revision:
            kwargs["revision"] = revision
        if spec.get("config"):
            ds = load_dataset(spec["hf_id"], spec["config"], **kwargs)
        else:
            ds = load_dataset(spec["hf_id"], **kwargs)

        out = cache_root / name
        ds.save_to_disk(str(out))
        meta = {
            "dataset": name,
            "hf_id": spec["hf_id"],
            "config": spec.get("config"),
            "split": spec["split"],
            "requested_revision": revision,
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
