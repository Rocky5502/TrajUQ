#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from trajuq.io import read_jsonl
from trajuq.schema import SCHEMA_VERSION


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw or scored TrajUQ JSONL artifacts.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--stage", choices=["raw", "scored", "auto"], default="auto")
    parser.add_argument("--require-revision", action="store_true")
    args = parser.parse_args()

    path = Path(args.run)
    rows = list(read_jsonl(path))
    ids: set[str] = set()
    failures: list[str] = []
    feature_names: set[str] | None = None
    models: set[str] = set()
    adapters: set[str] = set()
    schema_versions: set[str] = set()

    if not rows:
        failures.append("artifact contains zero records")

    if args.stage == "auto":
        has_labels = [r.get("correct") is not None for r in rows]
        stage = "scored" if rows and all(has_labels) else "raw"
        if any(has_labels) and not all(has_labels):
            failures.append("mixed scored/unscored rows")
    else:
        stage = args.stage

    for i, row in enumerate(rows):
        item_id = str(row.get("example_id", ""))
        if not item_id:
            failures.append(f"row {i}: missing example_id")
        elif item_id in ids:
            failures.append(f"row {i}: duplicate example_id={item_id}")
        ids.add(item_id)

        models.add(str(row.get("model")))
        adapters.add(str(row.get("adapter")))

        schema_version = row.get("schema_version")
        if not schema_version:
            failures.append(f"row {i}: missing schema_version")
        else:
            schema_versions.add(str(schema_version))
            if str(schema_version) != SCHEMA_VERSION:
                failures.append(
                    f"row {i}: schema_version={schema_version!r}; expected {SCHEMA_VERSION!r}"
                )

        feats = row.get("generation", {}).get("features", {})
        names = set(feats)
        if not feats:
            failures.append(f"row {i}: missing trajectory features")
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            failures.append(f"row {i}: feature schema mismatch")

        for key, value in feats.items():
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                failures.append(f"row {i}: non-finite feature {key}={value!r}")

        if row.get("reference") is not None:
            failures.append(f"row {i}: reference must never be stored in run artifacts")

        if stage == "raw" and row.get("correct") is not None:
            failures.append(f"row {i}: correctness leaked into raw artifact")
        if stage == "scored" and not isinstance(row.get("correct"), bool):
            failures.append(f"row {i}: scored artifact requires boolean correctness")

        if args.require_revision:
            revision = (
                row.get("metadata", {}).get("model_revision")
                or row.get("generation", {}).get("metadata", {}).get("model_revision")
            )
            if not revision:
                failures.append(f"row {i}: model revision is not pinned")

    audit = {
        "file": str(path),
        "stage": stage,
        "sha256": sha256_file(path),
        "records": len(rows),
        "unique_ids": len(ids),
        "models": sorted(models),
        "adapters": sorted(adapters),
        "schema_versions": sorted(schema_versions),
        "expected_schema_version": SCHEMA_VERSION,
        "feature_count": len(feature_names or set()),
        "feature_names": sorted(feature_names or set()),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }

    out = Path(args.output) if args.output else path.with_suffix(path.suffix + ".audit.json")
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
