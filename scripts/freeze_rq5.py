#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import yaml


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze prospective outcome-blind TMLR RQ5.")
    parser.add_argument("--config", default="configs/experiments/tmlr_rq5_blackbox.yaml")
    parser.add_argument("--schema", default="configs/schemas/rq5_api_provenance_v1.json")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--output", default="results/rq5/TMLR_RQ5_BLACKBOX_FREEZE.json")
    parser.add_argument("--prompt-ledger", default="results/rq5/prompt_render_ledger.jsonl")
    parser.add_argument("--cost-estimate", default="results/rq5/cost_estimate.json")
    args = parser.parse_args()

    output = Path(args.output)
    ledger_path = Path(args.prompt_ledger)
    if output.exists() or ledger_path.exists():
        raise SystemExit("Refusing to overwrite immutable RQ5 freeze evidence.")
    if git("status", "--porcelain"):
        raise SystemExit("RQ5 freeze requires a clean prospective source checkout.")

    config_path = Path(args.config)
    schema_path = Path(args.schema)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    provider_path = Path(config["provider"]["provider_snapshot"])
    inventory_path = Path(config["provider"]["inventory_snapshot"])
    pricing_path = Path(config["provider"]["pricing_contract"])
    cost_path = Path(args.cost_estimate)
    provider = json.loads(provider_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    pricing = json.loads(pricing_path.read_text(encoding="utf-8"))
    cost = json.loads(cost_path.read_text(encoding="utf-8"))

    if sha256_file(inventory_path) != provider["inventory_sha256"]:
        raise SystemExit("Authenticated model inventory hash drift.")
    available = {str(item.get("id")) for item in inventory.get("data", [])}
    models = list(config["models"])
    if any(model not in available for model in models):
        raise SystemExit("At least one frozen RQ5 target is absent from the authenticated inventory.")
    if sha256_file(Path(pricing["captured_page"])) != pricing["captured_page_sha256"]:
        raise SystemExit("Captured pricing-page hash drift.")
    if [row["model_id"] for row in pricing["models"]] != models:
        raise SystemExit("Pricing contract target order does not match frozen target order.")
    if not cost["within_hard_cap"] or float(cost["upper_total_rmb"]) > float(
        config["execution"]["hard_cost_cap_rmb"]
    ):
        raise SystemExit("Prospective RQ5 cost projection exceeds the hard cap.")

    system = str(config["prompt"]["system"])
    suffix = str(config["prompt"]["suffix"])
    template_contract = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": "{benchmark_prompt}" + suffix}]}
    template_sha = sha256_bytes(json.dumps(template_contract, sort_keys=True).encode())
    ledger_rows: list[dict] = []
    partitions: list[dict] = []
    for dataset in config["datasets"]:
        for split in ("calibration", "test"):
            path = Path(args.data_root) / dataset / f"{split}.jsonl"
            rows = read_jsonl(path)
            contract = config["partitions"]["assets"][dataset][split]
            observed_hash = sha256_file(path)
            if len(rows) != int(contract["rows"]) or observed_hash != str(contract["sha256"]):
                raise SystemExit(f"Frozen partition mismatch: {dataset}/{split}")
            start = len(ledger_rows)
            for row in rows:
                prompt = str(row["prompt"])
                request_messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt + suffix},
                ]
                ledger_rows.append({
                    "dataset": dataset,
                    "split": split,
                    "example_id": str(row["id"]),
                    "benchmark_prompt_sha256": sha256_bytes(prompt.encode()),
                    "request_messages_sha256": sha256_bytes(json.dumps(request_messages, sort_keys=True).encode()),
                })
            partitions.append({
                "dataset": dataset,
                "split": split,
                "rows": len(rows),
                "sha256": observed_hash,
                "ledger_start": start,
                "ledger_count": len(rows),
            })

    ledger_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger_rows)
    atomic_text(ledger_path, ledger_text)
    source_files = [
        config_path,
        schema_path,
        Path("scripts/capture_siliconflow_inventory.py"),
        Path("scripts/extract_siliconflow_pricing.py"),
        Path("scripts/estimate_rq5_cost.py"),
        Path("scripts/run_rq5_siliconflow.py"),
    ]
    manifest = {
        "schema": "trajuq_tmlr_rq5_blackbox_freeze_v1",
        "status": "PASS",
        "created_unix": int(time.time()),
        "original_authoritative_commit": config["original_authoritative_commit"],
        "prospective_rq5_source_commit": git("rev-parse", "HEAD"),
        "models": models,
        "provider_snapshot": {"path": str(provider_path).replace("\\", "/"), "sha256": sha256_file(provider_path)},
        "authenticated_inventory": {"path": str(inventory_path).replace("\\", "/"), "sha256": sha256_file(inventory_path)},
        "pricing_contract": {"path": str(pricing_path).replace("\\", "/"), "sha256": sha256_file(pricing_path)},
        "cost_estimate": {"path": str(cost_path).replace("\\", "/"), "sha256": sha256_file(cost_path), "expected_total_rmb": cost["expected_total_rmb"], "upper_total_rmb": cost["upper_total_rmb"], "hard_cost_cap_rmb": cost["hard_cost_cap_rmb"]},
        "prompt_template_sha256": template_sha,
        "prompt_ledger": str(ledger_path).replace("\\", "/"),
        "prompt_ledger_rows": len(ledger_rows),
        "prompt_ledger_sha256": sha256_bytes(ledger_text.encode()),
        "partitions": partitions,
        "decoding": config["decoding"],
        "execution": config["execution"],
        "signals": config["signals"],
        "analysis": config["analysis"],
        "source_files": [{"path": str(path).replace("\\", "/"), "sha256": sha256_file(path)} for path in source_files],
        "secrets_persisted": False,
        "outcomes_observed": False,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    atomic_text(output, manifest_text)
    atomic_text(output.with_suffix(output.suffix + ".sha256"), sha256_bytes(manifest_text.encode()) + "\n")
    print(f"PASS: RQ5 outcome-blind freeze -> {output}")


if __name__ == "__main__":
    main()
