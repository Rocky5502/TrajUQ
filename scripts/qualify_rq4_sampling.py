#!/usr/bin/env python
"""Run the frozen outcome-blind RQ4 sampling parity/throughput canary."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib import metadata as importlib_metadata
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from trajuq.adapters.llama_ar import LlamaARAdapter


EXPECTED_TORCH = "2.6.0+cu124"
EXPECTED_TRANSFORMERS = "4.52.4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify exact parity and throughput for frozen RQ4 sampling."
    )
    parser.add_argument(
        "--freeze-addendum",
        default="results/rq4/sampling_freeze_addendum.json",
    )
    parser.add_argument(
        "--raw-output",
        default="runs/rq4_sampling_canary/canary.raw.jsonl",
    )
    parser.add_argument(
        "--output",
        default="results/rq4/sampling_qualification.json",
    )
    args = parser.parse_args()

    output = Path(args.output)
    raw_output = Path(args.raw_output)
    if output.exists() or raw_output.exists():
        raise SystemExit("Refusing to overwrite immutable RQ4 sampling qualification evidence.")
    if git("status", "--porcelain"):
        raise SystemExit("RQ4 sampling qualification requires a clean source checkout.")

    freeze_path = Path(args.freeze_addendum)
    freeze_sidecar = freeze_path.with_suffix(freeze_path.suffix + ".sha256")
    freeze_sha = sha256_file(freeze_path)
    if freeze_sidecar.read_text(encoding="utf-8").strip() != freeze_sha:
        raise SystemExit("RQ4 sampling freeze sidecar mismatch.")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "PASS" or freeze.get("sampled_outcomes_observed") is not False:
        raise SystemExit("RQ4 sampling freeze is not an outcome-blind PASS artifact.")
    if freeze.get("source_commit") != git("rev-parse", "HEAD"):
        raise SystemExit("RQ4 sampling freeze source commit mismatch.")
    for item in freeze["source_files"]:
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise SystemExit(f"RQ4 sampling source drift: {item['path']}")

    settings = freeze["settings"]
    seed = int(settings["seeds"][0])
    canary_path = Path(freeze["canary_input"])
    if sha256_file(canary_path) != freeze["canary_input_sha256"]:
        raise SystemExit("RQ4 sampling canary input hash mismatch.")
    rows = [json.loads(line) for line in canary_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 12:
        raise SystemExit("RQ4 sampling canary must contain exactly 12 prompts.")

    import torch

    transformers_version = importlib_metadata.version("transformers")
    gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else ""
    runtime_pass = (
        torch.__version__ == EXPECTED_TORCH
        and transformers_version == EXPECTED_TRANSFORMERS
        and "A100" in gpu_name
    )
    if not runtime_pass:
        raise SystemExit(
            f"RQ4 runtime mismatch: torch={torch.__version__}, "
            f"transformers={transformers_version}, gpu={gpu_name!r}"
        )

    adapter = LlamaARAdapter(
        freeze["model_id"],
        device="cuda",
        dtype="bfloat16",
        revision=freeze["model_revision"],
    )
    output_rows: list[dict] = []
    parity_failures: list[dict] = []
    instrumented_seconds = 0.0
    reference_seconds = 0.0
    for row in rows:
        prompt = str(row["prompt"])
        torch.cuda.synchronize()
        started = time.perf_counter()
        native_ids = adapter.native_generate_token_ids(
            prompt,
            seed=seed,
            max_new_tokens=int(settings["max_new_tokens"]),
            do_sample=True,
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
            top_k=int(settings["top_k"]),
        )
        torch.cuda.synchronize()
        reference_seconds += time.perf_counter() - started

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        generated = adapter.generate(
            prompt,
            seed=seed,
            max_new_tokens=int(settings["max_new_tokens"]),
            do_sample=True,
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
            top_k=int(settings["top_k"]),
            top_k_telemetry=20,
        )
        torch.cuda.synchronize()
        wall_time = time.perf_counter() - started
        instrumented_seconds += wall_time

        if generated.token_ids != native_ids:
            parity_failures.append(
                {
                    "dataset": row["dataset"],
                    "example_id": row["id"],
                    "native_tokens": len(native_ids),
                    "instrumented_tokens": len(generated.token_ids),
                }
            )
        numeric_values = list(generated.features.values())
        if not numeric_values or not all(math.isfinite(float(value)) for value in numeric_values):
            raise SystemExit(f"Non-finite sampled telemetry: {row['dataset']}/{row['id']}")
        output_rows.append(
            {
                "schema_version": "trajuq-rq4-sampling-canary-v1",
                "example_id": str(row["id"]),
                "dataset": str(row["dataset"]),
                "split": str(row["split"]),
                "prompt_sha256": str(row["prompt_sha256"]),
                "model": freeze["model_id"],
                "model_revision": freeze["model_revision"],
                "seed": seed,
                "generation": asdict(generated),
                "correct": None,
                "reference": None,
                "metadata": {
                    "source_commit": freeze["source_commit"],
                    "source_clean": True,
                    "freeze_addendum_sha256": freeze_sha,
                    "wall_time_seconds": wall_time,
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                    "torch_version": torch.__version__,
                    "transformers_version": transformers_version,
                    "gpu_name": gpu_name,
                },
            }
        )

    raw_content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows)
    atomic_text(raw_output, raw_content)
    examples_per_hour = len(rows) * 3600.0 / instrumented_seconds
    expected_gpu_hours = float(freeze["expected_generations"]) / examples_per_hour
    status = "PASS" if not parity_failures else "FAIL"
    report = {
        "schema": "trajuq_tmlr_rq4_sampling_qualification_v1",
        "status": status,
        "go_no_go": "GO" if status == "PASS" else "NO_GO",
        "source_commit": freeze["source_commit"],
        "source_clean": True,
        "freeze_addendum": str(freeze_path),
        "freeze_addendum_sha256": freeze_sha,
        "raw_output": str(raw_output),
        "raw_output_sha256": sha256_file(raw_output),
        "outcome_blind": True,
        "canary_rows": len(rows),
        "seed": seed,
        "settings": settings,
        "exact_token_parity": not parity_failures,
        "parity_failures": parity_failures,
        "instrumented_wall_seconds": instrumented_seconds,
        "reference_wall_seconds": reference_seconds,
        "examples_per_hour": examples_per_hour,
        "expected_generations": freeze["expected_generations"],
        "projected_gpu_hours": expected_gpu_hours,
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "gpu": gpu_name,
        },
    }
    atomic_text(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
