#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from trajuq.io import read_jsonl


def unit_interval(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.30)
    parser.add_argument("--split-seed", default="2027")
    args = parser.parse_args()

    if not 0.0 < args.calibration_fraction < 1.0:
        raise SystemExit("--calibration-fraction must be in (0, 1)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_path = out_dir / "calibration.jsonl"
    test_path = out_dir / "test.jsonl"

    counts = {"calibration": 0, "test": 0}
    with cal_path.open("w", encoding="utf-8") as cal, test_path.open("w", encoding="utf-8") as test:
        for row in read_jsonl(args.input):
            key = f"{args.split_seed}:{row.get('dataset')}:{row['id']}"
            target = cal if unit_interval(key) < args.calibration_fraction else test
            split = "calibration" if target is cal else "test"
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            counts[split] += 1

    print(f"[TrajUQ] calibration={counts['calibration']} test={counts['test']}")


if __name__ == "__main__":
    main()
