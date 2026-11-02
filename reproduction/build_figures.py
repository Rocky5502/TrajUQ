#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    required = [
        "results/generated/secondary/rq2_sample_efficiency/rq2_cost_utility.pdf",
        "results/generated/secondary/rq5_heterogeneity/rq5_cell_forest.pdf",
        "results/generated/secondary/rq5_heterogeneity/rq5_coverage_risk.pdf",
    ]
    failures: list[str] = []
    for relative in required:
        path = ROOT / relative
        if not path.is_file() or not path.read_bytes().startswith(b"%PDF-"):
            failures.append(f"missing or invalid PDF: {relative}")
    print(json.dumps({"status": "PASS" if not failures else "FAIL", "checked": required, "failures": failures}, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
