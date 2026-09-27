from scripts.resident_generate_repeated_plan import (
    collect_jobs,
    parse_repeated_argv,
)


def repeated_argv(output_dir: str) -> list[str]:
    return [
        "python",
        "scripts/run_repeated_samples.py",
        "--adapter",
        "llada",
        "--model",
        "example/llada",
        "--revision",
        "model-sha",
        "--input",
        "data/input.jsonl",
        "--output-dir",
        output_dir,
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--seeds",
        "2027",
        "2028",
        "2029",
        "2030",
        "2031",
        "--temperature",
        "0.2",
        "--top-p",
        "1.0",
        "--max-new-tokens",
        "128",
        "--steps",
        "128",
        "--block-length",
        "128",
        "--llada-cfg-scale",
        "0.0",
        "--llada-remasking",
        "low_confidence",
        "--resume-verified",
        "--resume-partial",
    ]


def test_parse_repeated_argv_uses_shared_parser_contract():
    args = parse_repeated_argv(repeated_argv("runs/repeated/cal"))

    assert args.adapter == "llada"
    assert args.model == "example/llada"
    assert args.revision == "model-sha"
    assert args.seeds == [2027, 2028, 2029, 2030, 2031]
    assert args.temperature == 0.2
    assert args.device == "cuda"
    assert args.dtype == "bfloat16"
    assert args.resume_verified is True
    assert args.resume_partial is True


def test_parse_repeated_argv_rejects_non_resumable_plan():
    argv = repeated_argv("runs/repeated/cal")
    argv.remove("--resume-partial")

    try:
        parse_repeated_argv(argv)
    except ValueError as exc:
        assert "must enable verified and partial resume" in str(exc)
    else:
        raise AssertionError("resident RQ2 must reject non-resumable commands")


def test_collect_jobs_builds_two_partitions_per_frozen_cell():
    seeds = [2027, 2028, 2029, 2030, 2031]
    plan = {
        "cells": [
            {
                "cell": "llada:gsm8k",
                "adapter": "llada",
                "seeds": seeds,
                "calibration_generation_argv": repeated_argv(
                    "runs/repeated/llada/gsm8k/calibration"
                ),
                "test_generation_argv": repeated_argv(
                    "runs/repeated/llada/gsm8k/test"
                ),
            }
        ]
    }

    jobs = collect_jobs(plan)

    assert len(jobs) == 2
    assert [job["partition"] for job in jobs] == ["calibration", "test"]
    assert all(job["adapter"] == "llada" for job in jobs)
    assert all(job["seeds"] == seeds for job in jobs)
