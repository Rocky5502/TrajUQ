#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    required = [
        "results/generated/secondary/cross_regime_synthesis/regime_summary.csv",
        "results/generated/secondary/rq2_sample_efficiency/macro_metrics.csv",
        "results/generated/secondary/rq4_early_actionability/macro_metrics.csv",
        "results/generated/secondary/rq5_heterogeneity/cell_effects_display.csv",
        "results/generated/secondary/rq5_heterogeneity/component_comparison.csv",
    ]
    failures = [
        path
        for path in required
        if not (ROOT / path).is_file() or (ROOT / path).stat().st_size == 0
    ]
    rq5_path = ROOT / "results/generated/final/rq5_scientific_claim_registry.json"
    rq5 = json.loads(rq5_path.read_text(encoding="utf-8"))
    serialized = json.dumps(rq5, sort_keys=True)
    for token in (
        "-0.002990487543719062",
        "-0.015057308574727653",
        "0.010202548928115571",
        "0.6877312268773123",
        "NOT_SUPPORTED",
        "NOT_IDENTIFIABLE",
    ):
        if token not in serialized:
            failures.append(f"{rq5_path.relative_to(ROOT)} missing {token!r}")
    required.append(str(rq5_path.relative_to(ROOT)).replace("\\", "/"))
    print(json.dumps({"status": "PASS" if not failures else "FAIL", "checked": required, "failures": failures}, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
