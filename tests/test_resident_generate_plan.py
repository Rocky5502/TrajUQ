from scripts.resident_generate_plan import (
    collect_generation_jobs,
    generation_namespace,
)


def _argv(adapter: str, output: str = "runs/x.raw.jsonl") -> list[str]:
    common = [
        "python",
        "scripts/run_generation.py",
        "--adapter",
        adapter,
        "--model",
        f"example/{adapter}",
        "--revision",
        "model-sha",
        "--input",
        "data/input.jsonl",
        "--output",
        output,
        "--seed",
        "2027",
        "--max-new-tokens",
        "128",
    ]
    if adapter == "llada":
        return common + [
            "--steps",
            "128",
            "--block-length",
            "128",
            "--temperature",
            "0.0",
            "--llada-cfg-scale",
            "0.0",
            "--llada-remasking",
            "low_confidence",
        ]
    if adapter == "dream":
        return common + [
            "--steps",
            "128",
            "--temperature",
            "0.0",
            "--top-p",
            "1.0",
            "--dream-alg",
            "entropy",
            "--dream-alg-temp",
            "0.1",
        ]
    return common + [
        "--block-length",
        "4",
        "--temperature",
        "1.0",
        "--top-p",
        "1.0",
        "--sdar-denoising-steps",
        "4",
        "--sdar-top-k",
        "0",
        "--sdar-remasking-strategy",
        "low_confidence_dynamic",
        "--sdar-confidence-threshold",
        "0.85",
    ]


def test_generation_namespace_reconstructs_frozen_llada_settings():
    spec = {"generate_argv": _argv("llada")}
    args = generation_namespace(spec)

    assert args.adapter == "llada"
    assert args.model == "example/llada"
    assert args.revision == "model-sha"
    assert args.seed == 2027
    assert args.steps == 128
    assert args.block_length == 128
    assert args.temperature == 0.0
    assert args.llada_remasking == "low_confidence"
    assert args.device == "cuda"
    assert args.dtype == "bfloat16"


def test_generation_namespace_reconstructs_frozen_sdar_settings():
    spec = {"generate_argv": _argv("sdar")}
    args = generation_namespace(spec)

    assert args.adapter == "sdar"
    assert args.sdar_denoising_steps == 4
    assert args.block_length == 4
    assert args.temperature == 1.0
    assert args.sdar_confidence_threshold == 0.85


def test_collect_generation_jobs_primary_and_adaptive():
    base_spec = {
        "input": "data/input.jsonl",
        "raw_output": "runs/raw.jsonl",
        "generate_argv": _argv("dream"),
    }
    primary = {
        "cells": [
            {
                "cell": "dream:gsm8k",
                "adapter": "dream",
                "calibration": dict(base_spec),
                "test": {
                    **base_spec,
                    "raw_output": "runs/test.raw.jsonl",
                    "generate_argv": _argv("dream", "runs/test.raw.jsonl"),
                },
            }
        ]
    }
    adaptive = {
        "cells": [
            {
                "cell": "dream:gsm8k",
                "adapter": "dream",
                "calibration_base": dict(base_spec),
                "test_base": {
                    **base_spec,
                    "raw_output": "runs/base-test.raw.jsonl",
                    "generate_argv": _argv("dream", "runs/base-test.raw.jsonl"),
                },
            }
        ]
    }

    primary_jobs = collect_generation_jobs(primary, "primary")
    adaptive_jobs = collect_generation_jobs(adaptive, "adaptive")

    assert [(x["partition"], x["adapter"]) for x in primary_jobs] == [
        ("calibration", "dream"),
        ("test", "dream"),
    ]
    assert [(x["partition"], x["adapter"]) for x in adaptive_jobs] == [
        ("calibration", "dream"),
        ("test", "dream"),
    ]
