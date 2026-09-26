#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajuq.evaluation import selective_curve
from trajuq.io import read_jsonl


def reliability_points(y: np.ndarray, p: np.ndarray, bins: int):
    edges = np.linspace(0.0, 1.0, bins + 1)
    mean_p: list[float] = []
    mean_y: list[float] = []
    counts: list[int] = []
    for i in range(bins):
        left, right = edges[i], edges[i + 1]
        mask = (p >= left) & (p < right if i < bins - 1 else p <= right)
        if not np.any(mask):
            continue
        mean_p.append(float(p[mask].mean()))
        mean_y.append(float(y[mask].mean()))
        counts.append(int(mask.sum()))
    return np.asarray(mean_p), np.asarray(mean_y), counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render vector PDF risk-coverage and calibration figures from frozen predictions."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="paper/figures/generated")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit('Install analysis extras first: pip install -e ".[analysis]"') from exc

    rows = list(read_jsonl(args.predictions))
    if not rows:
        raise SystemExit("Prediction artifact is empty")

    ids = [str(r["example_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate example IDs in prediction artifact")

    y = np.asarray([int(r["y_error"]) for r in rows], dtype=int)
    p_base = np.asarray([float(r["p_error_decision_entropy"]) for r in rows], dtype=float)
    p_traj = np.asarray([float(r["p_error_trajuq"]) for r in rows], dtype=float)
    if not (np.all(np.isfinite(p_base)) and np.all(np.isfinite(p_traj))):
        raise SystemExit("Non-finite probabilities")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_curve = selective_curve(y, p_base)
    traj_curve = selective_curve(y, p_traj)

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(base_curve.coverage, base_curve.risk, label="Decision entropy")
    ax.plot(traj_curve.coverage, traj_curve.risk, label="TrajUQ")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    risk_path = out_dir / "risk_coverage.pdf"
    fig.savefig(risk_path, bbox_inches="tight")
    plt.close(fig)

    base_x, base_y, base_n = reliability_points(y, p_base, args.bins)
    traj_x, traj_y, traj_n = reliability_points(y, p_traj, args.bins)

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    ax.plot(base_x, base_y, marker="o", label="Decision entropy")
    ax.plot(traj_x, traj_y, marker="o", label="TrajUQ")
    ax.set_xlabel("Predicted error probability")
    ax.set_ylabel("Empirical error rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()
    calibration_path = out_dir / "calibration.pdf"
    fig.savefig(calibration_path, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "source_predictions": args.predictions,
        "n": len(rows),
        "bins": args.bins,
        "risk_coverage_pdf": str(risk_path),
        "calibration_pdf": str(calibration_path),
        "reliability_bin_counts": {
            "decision_entropy": base_n,
            "trajuq": traj_n,
        },
    }
    meta_path = out_dir / "figure_manifest.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
