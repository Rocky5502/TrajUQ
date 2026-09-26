#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajuq.io import read_jsonl
from trajuq.statistics import paired_bootstrap_improvement


def index(path: str) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    out = {str(row["example_id"]): row for row in rows}
    if len(out) != len(rows):
        raise SystemExit(f"Duplicate example IDs in {path}")
    if not out:
        raise SystemExit(f"Empty predictions file: {path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired held-out comparison of TrajUQ against repeated-sampling UQ."
    )
    parser.add_argument("--trajuq-predictions", required=True)
    parser.add_argument("--repeated-predictions", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    traj = index(args.trajuq_predictions)
    repeated = index(args.repeated_predictions)
    if set(traj) != set(repeated):
        raise SystemExit("TrajUQ/repeated-sampling prediction ID sets differ")

    ids = sorted(traj)
    y = np.asarray([int(traj[i]["y_error"]) for i in ids], dtype=int)
    repeated_y = np.asarray([int(repeated[i]["y_error"]) for i in ids], dtype=int)
    if not np.array_equal(y, repeated_y):
        raise SystemExit("Correctness labels differ between prediction artifacts")

    p_traj = np.asarray([float(traj[i]["p_error_trajuq"]) for i in ids], dtype=float)
    p_repeated = np.asarray(
        [float(repeated[i]["p_error_repeated_sampling"]) for i in ids],
        dtype=float,
    )
    if not np.all(np.isfinite(p_traj)) or not np.all(np.isfinite(p_repeated)):
        raise SystemExit("Non-finite prediction probabilities")

    payload = {
        "protocol": "trajuq_vs_repeated_sampling_v1",
        "comparison": "single_trajectory_trajuq_vs_five_sample_disagreement",
        "delta_direction": "positive_means_trajuq_is_better",
        "n": len(ids),
        "metrics": {
            metric: paired_bootstrap_improvement(
                y,
                p_traj,
                p_repeated,
                metric=metric,
                n_boot=args.n_boot,
                seed=args.seed,
            ).to_dict()
            for metric in ("auroc", "auprc", "aurc")
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
