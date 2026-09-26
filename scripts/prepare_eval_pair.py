#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from trajuq.datasets import normalize_row, split_bucket


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_rows(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def load_hf(load_dataset, spec: dict, split: str, revision: str | None):
    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    if spec.get("config"):
        return load_dataset(spec["hf_id"], spec["config"], **kwargs)
    return load_dataset(spec["hf_id"], **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare frozen calibration/test JSONL without tuning on the final test partition."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", default="data/manifests/evaluation_splits.json")
    parser.add_argument("--output-root", default="data/processed")
    parser.add_argument("--revision", default=None, help="Exact HF dataset revision/commit.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit('Install extras first: pip install -e ".[models]"') from exc

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.dataset not in manifest["datasets"]:
        raise SystemExit(f"Unknown dataset {args.dataset!r}")
    spec = manifest["datasets"][args.dataset]
    revision = args.revision or spec.get("revision")
    if spec.get("gated"):
        print("[TrajUQ] gated dataset: accept upstream terms and authenticate locally.")

    out_dir = Path(args.output_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_path = out_dir / "calibration.jsonl"
    test_path = out_dir / "test.jsonl"

    fingerprints: dict[str, str | None] = {}
    strategy = spec["strategy"]

    if strategy == "official":
        cal_ds = load_hf(load_dataset, spec, spec["calibration_split"], revision)
        test_ds = load_hf(load_dataset, spec, spec["test_split"], revision)
        fingerprints["calibration"] = getattr(cal_ds, "_fingerprint", None)
        fingerprints["test"] = getattr(test_ds, "_fingerprint", None)
        n_cal = write_rows(
            cal_path,
            (normalize_row(args.dataset, dict(row), i) for i, row in enumerate(cal_ds)),
        )
        n_test = write_rows(
            test_path,
            (normalize_row(args.dataset, dict(row), i) for i, row in enumerate(test_ds)),
        )

    elif strategy == "internal_holdout":
        source = load_hf(load_dataset, spec, spec["source_split"], revision)
        fingerprints["source"] = getattr(source, "_fingerprint", None)
        frac = float(spec["calibration_fraction"])
        seed = str(spec.get("split_seed", "2027"))
        cal_rows: list[dict] = []
        test_rows: list[dict] = []
        for i, row in enumerate(source):
            norm = normalize_row(args.dataset, dict(row), i)
            if split_bucket(args.dataset, str(norm["id"]), seed) < frac:
                cal_rows.append(norm)
            else:
                test_rows.append(norm)
        n_cal = write_rows(cal_path, cal_rows)
        n_test = write_rows(test_path, test_rows)
    else:
        raise SystemExit(f"Unknown split strategy: {strategy}")

    freeze = {
        "dataset": args.dataset,
        "hf_id": spec["hf_id"],
        "config": spec.get("config"),
        "strategy": strategy,
        "requested_revision": revision,
        "fingerprints": fingerprints,
        "n_calibration": n_cal,
        "n_test": n_test,
        "calibration_sha256": file_sha256(cal_path),
        "test_sha256": file_sha256(test_path),
        "nonstandard_note": spec.get("nonstandard_note"),
    }
    freeze_path = out_dir / "split_manifest.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[TrajUQ] calibration={n_cal} -> {cal_path}")
    print(f"[TrajUQ] test={n_test} -> {test_path}")
    print(f"[TrajUQ] split manifest -> {freeze_path}")


if __name__ == "__main__":
    main()
