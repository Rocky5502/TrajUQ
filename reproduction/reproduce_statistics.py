#!/usr/bin/env python
from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "estimate": -0.002990487543719062,
    "ci_95": [-0.015057308574727653, 0.010202548928115571],
    "holm_adjusted_p": 0.6877312268773123,
    "decision": "NOT_SUPPORTED",
}


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def same_number(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)


def main() -> int:
    manifests = [
        "results/generated/secondary/rq5_heterogeneity/manifest.json",
        "results/generated/secondary/rq5_missingness/manifest.json",
        "results/generated/secondary/rq5_leave_one_out/manifest.json",
    ]
    checks: list[dict[str, object]] = []
    failures: list[str] = []
    for relative in manifests:
        payload = load(relative)
        primary = payload.get("primary_rq5_immutable", {})
        ok = (
            payload.get("status") == "PASS"
            and primary.get("decision") == EXPECTED["decision"]
            and same_number(primary.get("estimate"), EXPECTED["estimate"])
            and all(same_number(a, b) for a, b in zip(primary.get("ci_95", []), EXPECTED["ci_95"]))
            and same_number(primary.get("holm_adjusted_p"), EXPECTED["holm_adjusted_p"])
        )
        checks.append({"path": relative, "status": "PASS" if ok else "FAIL"})
        if not ok:
            failures.append(f"immutable RQ5 result drift: {relative}")

    early = load("results/generated/secondary/rq4_early_actionability/manifest.json")
    early_ok = (
        early.get("status") == "PASS"
        and early.get("classification") == "SECONDARY_EXPLORATORY"
        and early.get("scientific_claim_authority") is False
    )
    checks.append({"path": "results/generated/secondary/rq4_early_actionability/manifest.json", "status": "PASS" if early_ok else "FAIL"})
    if not early_ok:
        failures.append("RQ4 early-actionability classification/status drift")

    report = {"status": "PASS" if not failures else "FAIL", "checks": checks, "failures": failures}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
