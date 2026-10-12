#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Outcome-blind RQ5 token/cost projection.")
    parser.add_argument("--config", default="configs/experiments/tmlr_rq5_blackbox.yaml")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--pricing", default="results/rq5/provider_snapshot/pricing_contract.json")
    parser.add_argument("--output", default="results/rq5/cost_estimate.json")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pricing = json.loads(Path(args.pricing).read_text(encoding="utf-8"))
    prompts = []
    for dataset in config["datasets"]:
        for split in ("calibration", "test"):
            prompts.extend(
                str(row["prompt"])
                for row in read_jsonl(Path(args.data_root) / dataset / f"{split}.jsonl")
            )
    repeats = int(config["decoding"]["repeated_samples"])
    fixed_chars = len(config["prompt"]["system"]) + len(config["prompt"]["suffix"])
    conservative_input_tokens_per_pass = sum(
        math.ceil((len(prompt) + fixed_chars) / 2.0) for prompt in prompts
    )
    expected_completion_tokens = 384
    maximum_completion_tokens = int(config["decoding"]["max_tokens"]) + int(
        config["decoding"]["thinking_budget"]
    )
    models = []
    total_expected = 0.0
    total_upper = 0.0
    for record in pricing["models"]:
        prices = {item["specification"]: float(item["price"]) for item in record["pricing"]}
        requests = len(prompts) * repeats
        input_tokens = conservative_input_tokens_per_pass * repeats
        expected_output = requests * expected_completion_tokens
        maximum_output = requests * maximum_completion_tokens
        input_cost = input_tokens / 1_000_000 * prices["prompt"]
        expected_cost = input_cost + expected_output / 1_000_000 * prices["completion"]
        upper_cost = input_cost + maximum_output / 1_000_000 * prices["completion"]
        total_expected += expected_cost
        total_upper += upper_cost
        models.append(
            {
                "model_id": record["model_id"],
                "requests": requests,
                "conservative_input_tokens": input_tokens,
                "expected_completion_tokens": expected_output,
                "maximum_completion_plus_reasoning_tokens": maximum_output,
                "expected_cost_rmb": expected_cost,
                "upper_cost_rmb": upper_cost,
            }
        )
    payload = {
        "schema": "trajuq_rq5_cost_estimate_v1",
        "examples_per_model": len(prompts),
        "repeats": repeats,
        "requests_total": len(prompts) * repeats * len(models),
        "token_estimation": "ceil_unicode_characters_divided_by_two",
        "expected_completion_tokens_per_request": expected_completion_tokens,
        "maximum_completion_plus_reasoning_tokens_per_request": maximum_completion_tokens,
        "models": models,
        "expected_total_rmb": total_expected,
        "upper_total_rmb": total_upper,
        "hard_cost_cap_rmb": float(config["execution"]["hard_cost_cap_rmb"]),
        "within_hard_cap": total_upper <= float(config["execution"]["hard_cost_cap_rmb"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
