#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajuq.datasets import normalize_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legacy single-cache normalizer. Prefer prepare_eval_pair.py for primary experiments."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        from datasets import load_from_disk
    except ImportError as e:
        raise SystemExit('Install extras first: pip install -e ".[models]"') from e

    ds = load_from_disk(str(Path(args.cache_dir) / args.dataset))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.dataset}.jsonl"

    count = min(len(ds), args.limit) if args.limit is not None else len(ds)
    with out.open("w", encoding="utf-8") as f:
        for idx in range(count):
            rec = normalize_row(args.dataset, dict(ds[idx]), idx)
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"[TrajUQ] wrote {count} normalized records to {out}")


if __name__ == "__main__":
    main()
