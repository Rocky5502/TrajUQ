#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from trajuq.io import append_jsonl, sha256_text
from trajuq.schema import ExampleRecord, GenerationResult, StepSummary


def make_features(is_error: bool, rng: random.Random) -> dict[str, float]:
    e = 1.0 if is_error else 0.0
    jitter = lambda scale: rng.uniform(-scale, scale)
    entropy_mean = 0.28 + 0.28 * e + jitter(0.04)
    entropy_p90 = 0.34 + 0.35 * e + jitter(0.04)
    entropy_final = 0.32 + 0.12 * e + jitter(0.08)  # intentionally weaker baseline
    margin_mean = 0.72 - 0.40 * e + jitter(0.04)
    revision = 0.04 + 0.42 * e + jitter(0.025)
    late_revision = 0.02 + 0.48 * e + jitter(0.02)
    js_mean = 0.04 + 0.28 * e + jitter(0.015)
    js_p90 = 0.06 + 0.34 * e + jitter(0.015)
    mask_auc = 0.20 + 0.42 * e + jitter(0.03)
    late_mask = 0.03 + 0.40 * e + jitter(0.02)
    commit = 0.18 + 0.58 * e + jitter(0.03)

    return {
        "trajectory_steps": 8.0,
        "entropy_mean": max(0.0, entropy_mean),
        "entropy_p90": max(0.0, entropy_p90),
        "entropy_max": max(0.0, entropy_p90 + 0.06),
        "entropy_final": max(0.0, entropy_final),
        "entropy_slope": 0.04 * e - 0.10 * (1.0 - e) + jitter(0.02),
        "confidence_mean": min(1.0, max(0.0, 0.78 - 0.42 * e + jitter(0.03))),
        "confidence_final": min(1.0, max(0.0, 0.74 - 0.18 * e + jitter(0.08))),
        "margin_mean": min(1.0, max(0.0, margin_mean)),
        "margin_min": min(1.0, max(0.0, margin_mean - 0.12)),
        "margin_final": min(1.0, max(0.0, margin_mean + jitter(0.05))),
        "revision_rate": max(0.0, revision),
        "late_revision_rate": max(0.0, late_revision),
        "revision_slope": 0.18 * e + jitter(0.02),
        "js_mean": max(0.0, js_mean),
        "js_p90": max(0.0, js_p90),
        "js_max": max(0.0, js_p90 + 0.04),
        "js_final": max(0.0, js_mean + jitter(0.02)),
        "js_slope": 0.12 * e + jitter(0.02),
        "late_js_mean": max(0.0, js_mean + 0.08 * e + jitter(0.01)),
        "mask_auc": min(1.0, max(0.0, mask_auc)),
        "mask_fraction_final": 0.0,
        "late_mask_fraction": min(1.0, max(0.0, late_mask)),
        "commitment_time_mean": min(1.0, max(0.0, commit)),
        "commitment_time_p90": min(1.0, max(0.0, commit + 0.08)),
    }


def write_partition(root: Path, name: str, start: int, count: int, seed: int) -> tuple[Path, Path]:
    rng = random.Random(f"{seed}:{name}")
    input_path = root / f"{name}.input.jsonl"
    raw_path = root / f"{name}.raw.jsonl"

    for path in (input_path, raw_path):
        if path.exists():
            path.unlink()

    for offset in range(count):
        i = start + offset
        prompt = f"Synthetic arithmetic item {i}: return the integer {i + 10}."
        reference = str(i + 10)
        is_error = (i % 3 == 0)
        output = str(i + 11) if is_error else reference
        features = make_features(is_error, rng)

        source = {
            "id": f"syn-{i}",
            "dataset": "synthetic",
            "prompt": prompt,
            "reference": reference,
            "metadata": {"synthetic": True},
        }
        append_jsonl(input_path, source)

        step = StepSummary(
            step=7,
            entropy_mean=features["entropy_final"],
            entropy_p90=features["entropy_p90"],
            confidence_mean=features["confidence_final"],
            margin_mean=features["margin_final"],
            mask_fraction=0.0,
            revision_fraction=features["late_revision_rate"],
        )
        generation = GenerationResult(
            text=f"FINAL: {output}",
            token_ids=[],
            features=features,
            steps=[step],
            metadata={
                "model_id": "toy/diffusion",
                "model_revision": "synthetic-v1",
                "steps_requested": 8,
            },
        )
        record = ExampleRecord(
            example_id=f"syn-{i}",
            dataset="synthetic",
            prompt_sha256=sha256_text(prompt),
            model="toy/diffusion",
            adapter="synthetic",
            seed=seed,
            generation=generation,
            correct=None,
            reference=None,
            metadata={"model_revision": "synthetic-v1", "synthetic": True},
        )
        append_jsonl(raw_path, record.to_dict())

    return input_path, raw_path


def run(*args: str) -> None:
    cmd = [sys.executable, *args]
    print("[TrajUQ smoke]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/smoke_e2e")
    parser.add_argument("--calibration", type=int, default=48)
    parser.add_argument("--test", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    cal_input, cal_raw = write_partition(root, "calibration", 0, args.calibration, args.seed)
    test_input, test_raw = write_partition(root, "test", 10000, args.test, args.seed)

    cal_scored = root / "calibration.scored.jsonl"
    test_scored = root / "test.scored.jsonl"
    metrics = root / "metrics.json"
    for path in (cal_scored, test_scored, metrics):
        if path.exists():
            path.unlink()

    for raw in (cal_raw, test_raw):
        run("scripts/audit_run.py", "--run", str(raw), "--stage", "raw", "--require-revision")

    run(
        "scripts/score_outputs.py",
        "--run", str(cal_raw),
        "--input", str(cal_input),
        "--output", str(cal_scored),
    )
    run(
        "scripts/score_outputs.py",
        "--run", str(test_raw),
        "--input", str(test_input),
        "--output", str(test_scored),
    )

    for scored in (cal_scored, test_scored):
        run("scripts/audit_run.py", "--run", str(scored), "--stage", "scored", "--require-revision")

    run(
        "scripts/evaluate_run.py",
        "--calibration", str(cal_scored),
        "--test", str(test_scored),
        "--output", str(metrics),
    )

    result = json.loads(metrics.read_text(encoding="utf-8"))
    trajectory_auroc = float(result["trajectory_logistic"]["auroc"])
    if trajectory_auroc < 0.90:
        raise SystemExit(f"synthetic smoke regression: trajectory AUROC={trajectory_auroc:.3f}")

    print(f"[TrajUQ] END-TO-END SMOKE PASS -> {root}")


if __name__ == "__main__":
    main()
