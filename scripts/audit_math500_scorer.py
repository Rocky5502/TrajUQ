#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from importlib import metadata
from pathlib import Path

from trajuq.scoring import math500_equivalent, score_math500


EQUIVALENT_CASES = [
    (r"FINAL: \frac{1}{2}", r"\frac{1}{2}"),
    (r"FINAL: \frac{2}{4}", r"\frac{1}{2}"),
    (r"FINAL: 0.5", r"\frac{1}{2}"),
    (r"FINAL: 2", r"2"),
    (r"FINAL: -3", r"-3"),
]

def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


EXPECTED_MATH_VERIFY_VERSION = "0.9.0"
EXPECTED_ANTLR_VERSION = "4.13.2"


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


NON_EQUIVALENT_CASES = [
    (r"FINAL: \frac{1}{2}", r"\frac{2}{3}"),
    (r"FINAL: 2", r"3"),
    (r"FINAL: -3", r"3"),
    (r"FINAL: 0", r"1"),
    (r"FINAL: 10", r"100"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic pre-experiment audit of the pinned MATH-500 scorer."
    )
    parser.add_argument(
        "--output",
        default="results/math500_scorer_audit.json",
    )
    args = parser.parse_args()

    results: list[dict] = []
    for prediction, reference in EQUIVALENT_CASES:
        observed = score_math500(prediction, reference)
        results.append(
            {
                "kind": "equivalent",
                "prediction": prediction,
                "reference": reference,
                "expected": True,
                "observed": bool(observed),
            }
        )
    for prediction, reference in NON_EQUIVALENT_CASES:
        observed = score_math500(prediction, reference)
        results.append(
            {
                "kind": "non_equivalent",
                "prediction": prediction,
                "reference": reference,
                "expected": False,
                "observed": bool(observed),
            }
        )

    pairwise_checks = [
        {
            "left": r"FINAL: \frac{2}{4}",
            "right": r"FINAL: 0.5",
            "expected": True,
            "observed": bool(
                math500_equivalent(
                    r"FINAL: \frac{2}{4}",
                    r"FINAL: 0.5",
                )
            ),
        },
        {
            "left": r"FINAL: \frac{1}{2}",
            "right": r"FINAL: \frac{3}{4}",
            "expected": False,
            "observed": bool(
                math500_equivalent(
                    r"FINAL: \frac{1}{2}",
                    r"FINAL: \frac{3}{4}",
                )
            ),
        },
    ]

    failures = [
        item
        for item in results + pairwise_checks
        if bool(item["observed"]) != bool(item["expected"])
    ]

    observed_math_verify = package_version("math-verify")
    observed_antlr = package_version("antlr4-python3-runtime")
    if observed_math_verify != EXPECTED_MATH_VERIFY_VERSION:
        failures.append(
            {
                "kind": "dependency",
                "package": "math-verify",
                "expected": EXPECTED_MATH_VERIFY_VERSION,
                "observed": observed_math_verify,
            }
        )
    if observed_antlr != EXPECTED_ANTLR_VERSION:
        failures.append(
            {
                "kind": "dependency",
                "package": "antlr4-python3-runtime",
                "expected": EXPECTED_ANTLR_VERSION,
                "observed": observed_antlr,
            }
        )

    commit = git_head()
    if not commit:
        failures.append({"kind": "environment", "reason": "git_commit_unavailable"})

    payload = {
        "protocol": "trajuq_math500_scorer_audit_v2",
        "git_commit": commit,
        "dependency": "math-verify[antlr4_13_2]==0.9.0",
        "dependency_versions": {
            "math-verify": observed_math_verify,
            "antlr4-python3-runtime": observed_antlr,
        },
        "expected_dependency_versions": {
            "math-verify": EXPECTED_MATH_VERIFY_VERSION,
            "antlr4-python3-runtime": EXPECTED_ANTLR_VERSION,
        },
        "status": "PASS" if not failures else "FAIL",
        "scorer_cases": results,
        "pairwise_clustering_cases": pairwise_checks,
        "failure_count": len(failures),
        "failures": failures,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
