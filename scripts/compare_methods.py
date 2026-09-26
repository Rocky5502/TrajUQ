#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import numpy as np

from trajuq.io import read_jsonl
from trajuq.risk import RAW_RISK_VERSION, raw_trajectory_risk
from trajuq.statistics import paired_bootstrap_difference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap comparison of training-free TrajUQ risk vs final entropy."
    )
    parser.add_argument("--input", required=True, help="Scored test JSONL")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    rows = [r for r in read_jsonl(args.input) if r.get("correct") is not None]
    if not rows:
        raise SystemExit("No scored rows")

    y = np.asarray([0 if bool(r["correct"]) else 1 for r in rows], dtype=int)
    trajectory = np.asarray(
        [raw_trajectory_risk(r["generation"]["features"]) for r in rows],
        dtype=float,
    )
    final_entropy = np.asarray(
        [float(r["generation"]["features"]["entropy_final"]) for r in rows],
        dtype=float,
    )

    payload = {
        "risk_version": RAW_RISK_VERSION,
        "n": len(rows),
        "comparison": "raw_trajectory_risk_minus_final_entropy",
        "metrics": {},
    }
    for metric in ["auroc", "auprc", "aurc"]:
        payload["metrics"][metric] = paired_bootstrap_difference(
            y,
            trajectory,
            final_entropy,
            metric=metric,
            n_boot=args.n_boot,
            seed=args.seed,
        ).to_dict()

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
