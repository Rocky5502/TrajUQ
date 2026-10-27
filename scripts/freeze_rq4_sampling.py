#!/usr/bin/env python
"""Freeze the pre-registered RQ4 five-sample execution addendum.

This addendum does not change the frozen RQ4 scientific settings. It binds the
sampling implementation and an outcome-blind 12-prompt qualification set before
any sampled RQ4 output is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import yaml


EXPECTED_ORIGINAL_FREEZE_SHA256 = (
    "4f002417c736a7b3b7859e231fbf9cd0458f666d02e087cb7a364c56461093aa"
)
EXPECTED_SETTINGS = {
    "seeds": [2027, 2028, 2029, 2030, 2031],
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 0,
    "max_new_tokens": 512,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the outcome-blind RQ4 sampled-generation qualification."
    )
    parser.add_argument("--config", default="configs/experiments/tmlr_rq4.yaml")
    parser.add_argument("--model-config", default="configs/models/llama31_ar.yaml")
    parser.add_argument(
        "--telemetry-schema",
        default="configs/schemas/rq4_ar_telemetry_v1.json",
    )
    parser.add_argument("--original-freeze", required=True)
    parser.add_argument("--prompt-ledger", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--greedy-artifact-root", required=True)
    parser.add_argument(
        "--sampled-artifact-root",
        default="runs/rq4/llama31_five_sample",
    )
    parser.add_argument(
        "--output",
        default="results/rq4/sampling_freeze_addendum.json",
    )
    parser.add_argument(
        "--canary-input",
        default="results/rq4/sampling_canary_input.jsonl",
    )
    args = parser.parse_args()

    output = Path(args.output)
    canary_input = Path(args.canary_input)
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise SystemExit("Refusing to overwrite immutable RQ4 sampling freeze evidence.")
    if canary_input.exists():
        raise SystemExit("Refusing to overwrite immutable RQ4 sampling canary input.")
    if git("status", "--porcelain"):
        raise SystemExit("RQ4 sampling freeze requires a clean source checkout.")

    sampled_root = Path(args.sampled_artifact_root)
    if sampled_root.exists() and any(path.is_file() for path in sampled_root.rglob("*")):
        raise SystemExit("Sampled RQ4 artifacts already exist; prospective freeze is forbidden.")

    config_path = Path(args.config)
    model_config_path = Path(args.model_config)
    telemetry_schema_path = Path(args.telemetry_schema)
    original_freeze_path = Path(args.original_freeze)
    prompt_ledger_path = Path(args.prompt_ledger)
    data_root = Path(args.data_root)
    greedy_root = Path(args.greedy_artifact_root)

    original_freeze_sha = sha256_file(original_freeze_path)
    if original_freeze_sha != EXPECTED_ORIGINAL_FREEZE_SHA256:
        raise SystemExit("Original RQ4 freeze hash mismatch.")
    original = json.loads(original_freeze_path.read_text(encoding="utf-8"))
    if original.get("status") != "PASS" or original.get("outcomes_observed") is not False:
        raise SystemExit("Original RQ4 freeze is not the required outcome-blind PASS artifact.")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    settings = config["generation"]["five_sample_baseline"]
    if settings != EXPECTED_SETTINGS:
        raise SystemExit("Pre-registered RQ4 five-sample settings changed.")
    if int(config["generation"]["outcome_blind_canary_rows_per_dataset"]) != 2:
        raise SystemExit("RQ4 canary size changed from two rows per dataset.")
    if sha256_file(config_path) != original["experiment_config_sha256"]:
        raise SystemExit("RQ4 experiment configuration drifted from the original freeze.")
    if sha256_file(model_config_path) != original["model_config_sha256"]:
        raise SystemExit("RQ4 model configuration drifted from the original freeze.")
    if sha256_file(telemetry_schema_path) != original["telemetry_schema_sha256"]:
        raise SystemExit("RQ4 telemetry schema drifted from the original freeze.")
    if sha256_file(prompt_ledger_path) != original["prompt_ledger_sha256"]:
        raise SystemExit("RQ4 prompt ledger drifted from the original freeze.")

    ledger = read_jsonl(prompt_ledger_path)
    ledger_by_key = {
        (row["dataset"], row["split"], str(row["example_id"])): row for row in ledger
    }
    if len(ledger_by_key) != len(ledger) or len(ledger) != 3884:
        raise SystemExit("RQ4 prompt ledger is incomplete or contains duplicate rows.")

    canary_rows: list[dict] = []
    total_rows = 0
    for dataset in config["datasets"]:
        for split in ("calibration", "test"):
            source = data_root / dataset / f"{split}.jsonl"
            rows = read_jsonl(source)
            contract = config["partitions"]["assets"][dataset][split]
            if len(rows) != int(contract["rows"]) or sha256_file(source) != contract["sha256"]:
                raise SystemExit(f"Frozen partition mismatch: {dataset}/{split}")
            total_rows += len(rows)
        calibration = read_jsonl(data_root / dataset / "calibration.jsonl")
        for row in calibration[:2]:
            example_id = str(row["id"])
            prompt = str(row["prompt"])
            ledger_row = ledger_by_key[(dataset, "calibration", example_id)]
            if sha256_text(prompt) != ledger_row["prompt_sha256"]:
                raise SystemExit(f"Canary prompt mismatch: {dataset}/{example_id}")
            canary_rows.append(
                {
                    "id": example_id,
                    "dataset": dataset,
                    "split": "calibration",
                    "prompt": prompt,
                    "prompt_sha256": ledger_row["prompt_sha256"],
                    "rendered_prompt_sha256": ledger_row["rendered_prompt_sha256"],
                }
            )
    if total_rows != 3884 or len(canary_rows) != 12:
        raise SystemExit("RQ4 workload or canary cardinality mismatch.")

    greedy_files = sorted(greedy_root.glob("*/*.raw.jsonl"))
    greedy_rows = sum(len(read_jsonl(path)) for path in greedy_files)
    if len(greedy_files) != 12 or greedy_rows != 3884:
        raise SystemExit("Pre-existing greedy RQ4 evidence is incomplete.")

    canary_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in canary_rows)
    atomic_text(canary_input, canary_text)
    source_files = [
        config_path,
        model_config_path,
        telemetry_schema_path,
        Path("src/trajuq/adapters/llama_ar.py"),
        Path("scripts/run_generation.py"),
        Path("scripts/freeze_rq4_sampling.py"),
        Path("scripts/qualify_rq4_sampling.py"),
    ]
    manifest = {
        "schema": "trajuq_tmlr_rq4_sampling_addendum_v1",
        "status": "PASS",
        "created_unix": int(time.time()),
        "source_commit": git("rev-parse", "HEAD"),
        "source_clean": True,
        "original_freeze_path": str(original_freeze_path),
        "original_freeze_sha256": original_freeze_sha,
        "original_rq4_source_commit": original["prospective_rq4_source_commit"],
        "original_authoritative_commit": original["original_authoritative_commit"],
        "model_id": model_config["model_id"],
        "model_revision": model_config["revision"],
        "scientific_settings_changed": False,
        "amendment_scope": "execution_support_for_pre_registered_five_sample_baseline",
        "preexisting_greedy_outcomes_acknowledged": True,
        "preexisting_greedy_rows": greedy_rows,
        "sampled_outcomes_observed": False,
        "settings": settings,
        "prompt_rows": total_rows,
        "expected_generations": total_rows * len(settings["seeds"]),
        "canary_selection": "first_two_frozen_calibration_rows_per_dataset_in_config_order",
        "canary_rows": len(canary_rows),
        "canary_input": str(canary_input),
        "canary_input_sha256": sha256_text(canary_text),
        "source_files": [
            {"path": path.as_posix(), "sha256": sha256_file(path)} for path in source_files
        ],
    }
    if manifest["expected_generations"] != 19420:
        raise SystemExit("RQ4 expected generation count is not exactly 19,420.")
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    atomic_text(output, content)
    atomic_text(output.with_suffix(output.suffix + ".sha256"), sha256_text(content) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
