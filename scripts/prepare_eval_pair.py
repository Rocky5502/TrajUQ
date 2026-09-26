#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

from trajuq.data_integrity import audit_split_pair
from trajuq.datasets import (
    NORMALIZER_VERSION,
    namespace_record_id,
    normalize_row,
    split_bucket,
)


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
    parser.add_argument(
        "--allow-unpinned-revision",
        action="store_true",
        help="Exploratory use only; primary experiments must pin a revision.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing prepared split. Never use after outcome inspection.",
    )
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
    if not revision and not args.allow_unpinned_revision:
        raise SystemExit(
            "Dataset revision is unpinned. Resolve an immutable Hugging Face SHA "
            "and pass --revision. Use --allow-unpinned-revision only for exploration."
        )
    if spec.get("gated"):
        print("[TrajUQ] gated dataset: accept upstream terms and authenticate locally.")

    out_dir = Path(args.output_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    cal_path = out_dir / "calibration.jsonl"
    test_path = out_dir / "test.jsonl"
    freeze_path = out_dir / "split_manifest.json"

    audit_path = out_dir / "split_audit.json"
    existing = [
        path
        for path in (cal_path, test_path, freeze_path, audit_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise SystemExit(
            "Refusing to replace frozen split artifacts: "
            + ", ".join(str(path) for path in existing)
            + ". Use a new output root, or --overwrite only before outcomes are inspected."
        )

    fingerprints: dict[str, str | None] = {}
    strategy = spec["strategy"]

    if strategy == "official":
        cal_ds = load_hf(load_dataset, spec, spec["calibration_split"], revision)
        test_ds = load_hf(load_dataset, spec, spec["test_split"], revision)
        fingerprints["calibration"] = getattr(cal_ds, "_fingerprint", None)
        fingerprints["test"] = getattr(test_ds, "_fingerprint", None)
        def official_rows(ds, split_name):
            for i, row in enumerate(ds):
                norm = normalize_row(args.dataset, dict(row), i)
                norm["id"] = namespace_record_id(
                    args.dataset,
                    split_name,
                    str(norm["id"]),
                )
                norm.setdefault("metadata", {})["source_split"] = split_name
                yield norm

        n_cal = write_rows(
            cal_path,
            official_rows(cal_ds, spec["calibration_split"]),
        )
        n_test = write_rows(
            test_path,
            official_rows(test_ds, spec["test_split"]),
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
            raw_id = str(norm["id"])
            norm["id"] = namespace_record_id(
                args.dataset,
                spec["source_split"],
                raw_id,
            )
            norm.setdefault("metadata", {})["source_split"] = spec["source_split"]
            if split_bucket(args.dataset, raw_id, seed) < frac:
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
        "normalizer_version": NORMALIZER_VERSION,
        "requested_revision": revision,
        "fingerprints": fingerprints,
        "n_calibration": n_cal,
        "n_test": n_test,
        "calibration_sha256": file_sha256(cal_path),
        "test_sha256": file_sha256(test_path),
        "nonstandard_note": spec.get("nonstandard_note"),
    }
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    audit = audit_split_pair(cal_path, test_path, freeze_path)
    audit_path = out_dir / "split_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if audit["status"] != "PASS":
        raise SystemExit(
            f"Prepared split failed integrity audit; inspect {audit_path}"
        )

    print(f"[TrajUQ] calibration={n_cal} -> {cal_path}")
    print(f"[TrajUQ] test={n_test} -> {test_path}")
    print(f"[TrajUQ] split manifest -> {freeze_path}")
    print(f"[TrajUQ] split audit -> {audit_path}")


if __name__ == "__main__":
    main()
