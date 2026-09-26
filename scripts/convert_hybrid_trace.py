#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from trajuq.hybrid_trace import HYBRID_TRACE_SCHEMA_VERSION, summarize_hybrid_trace
from trajuq.io import read_jsonl


UPSTREAM_REPOSITORY = "yuchen-zhu-zyc/HybridDiffusion"
UPSTREAM_COMMIT = "6ca547aebb72bfe897e80e0a683c776789e5f38c"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HybridDiffusion's native AR-Trust JSONL trace into TrajUQ diagnostics."
    )
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="yuchen-zhu-zyc/HybridDiffusion-2B")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--late-fraction", type=float, default=0.25)
    parser.add_argument("--require-topk", type=int, default=5)
    args = parser.parse_args()

    trace_path = Path(args.trace)
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output_path}")

    records = list(read_jsonl(trace_path))
    if not records:
        raise SystemExit("Trace file is empty")

    topk_lengths = [
        len(row["top_logits"])
        for row in records
        if isinstance(row.get("top_logits"), list)
    ]
    if not topk_lengths:
        raise SystemExit(
            "Native trace contains no top_logits. Set "
            "SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TRACE_TOPK to a positive value."
        )
    if min(topk_lengths) < args.require_topk:
        raise SystemExit(
            f"Trace top-k is too small: min={min(topk_lengths)}, required={args.require_topk}"
        )

    summaries = summarize_hybrid_trace(records, late_fraction=args.late_fraction)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_hash = sha256_file(trace_path)

    with output_path.open("w", encoding="utf-8") as f:
        for summary in summaries:
            payload = summary.to_dict()
            payload["metadata"].update(
                {
                    "model_id": args.model,
                    "model_revision": args.model_revision,
                    "upstream_repository": UPSTREAM_REPOSITORY,
                    "upstream_commit": UPSTREAM_COMMIT,
                    "source_trace_sha256": trace_hash,
                    "source_trace_path": str(trace_path),
                    "schema_version": HYBRID_TRACE_SCHEMA_VERSION,
                }
            )
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "status": "PASS",
                "requests": len(summaries),
                "trace_rows": len(records),
                "trace_sha256": trace_hash,
                "output": str(output_path),
                "warning": (
                    "topk_entropy_proxy is not full-vocabulary entropy; "
                    "use it only as a HybridDiffusion-native diagnostic"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
