#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import yaml


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a live RQ5 cost ledger without secrets.")
    parser.add_argument("--config", default="configs/experiments/tmlr_rq5_blackbox.yaml")
    parser.add_argument("--freeze", default="results/rq5/TMLR_RQ5_BLACKBOX_FREEZE.json")
    parser.add_argument("--authorization", default="configs/operations/rq5_budget_authorization_af0347a.yaml")
    parser.add_argument("--records", default="results/rq5/api/records")
    parser.add_argument("--output", default="results/rq5/api/cost_ledger.json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    authorization = yaml.safe_load(Path(args.authorization).read_text(encoding="utf-8"))
    freeze_path = Path(args.freeze)
    freeze_sha = sha256_bytes(freeze_path.read_bytes())
    if freeze_sha != authorization["freeze_sha256"]:
        raise SystemExit("Budget authorization is not bound to the active RQ5 freeze.")

    expected: dict[tuple[str, str], int] = {}
    repeats = int(config["decoding"]["repeated_samples"])
    for dataset in config["datasets"]:
        rows = sum(int(config["partitions"]["assets"][dataset][split]["rows"]) for split in ("calibration", "test"))
        for model in config["models"]:
            expected[(model, dataset)] = rows * repeats

    completed = defaultdict(int)
    costs = defaultdict(float)
    model_costs = defaultdict(float)
    model_completed = defaultdict(int)
    total_cost = 0.0
    for path in Path(args.records).glob("*/*/*/*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("freeze_sha256") != freeze_sha:
            raise SystemExit(f"Freeze hash drift in record: {path}")
        key = (str(record["requested_model"]), str(record["dataset"]))
        cost = float(record.get("estimated_cost_rmb", 0.0))
        completed[key] += 1
        costs[key] += cost
        model_completed[key[0]] += 1
        model_costs[key[0]] += cost
        total_cost += cost

    fallback_by_model = {}
    estimate = json.loads(Path("results/rq5/cost_estimate.json").read_text(encoding="utf-8"))
    for row in estimate["models"]:
        fallback_by_model[row["model_id"]] = float(row["expected_cost_rmb"]) / int(row["requests"])

    cells = []
    projected_total = total_cost
    for model in config["models"]:
        for dataset in config["datasets"]:
            key = (model, dataset)
            done = completed[key]
            target = expected[key]
            if done:
                mean_cost = costs[key] / done
            elif model_completed[model]:
                mean_cost = model_costs[model] / model_completed[model]
            else:
                mean_cost = fallback_by_model[model]
            projected = costs[key] + (target - done) * mean_cost
            projected_total += (target - done) * mean_cost
            cells.append({
                "model": model,
                "dataset": dataset,
                "completed_requests": done,
                "expected_requests": target,
                "cumulative_cost_rmb": costs[key],
                "mean_cost_per_completed_request_rmb": mean_cost,
                "projected_cell_completion_cost_rmb": projected,
            })

    preferred = float(authorization["preferred_target_rmb"])
    warning = float(authorization["warning_threshold_rmb"])
    hard = float(authorization["absolute_hard_stop_rmb"])
    if total_cost >= hard:
        mode = "HARD_STOP"
    elif total_cost >= warning:
        mode = "STRICT_COST_WATCH"
    elif total_cost >= preferred:
        mode = "CONTINUE_ONLY_IF_PROJECTED_LTE_HARD_STOP"
    else:
        mode = "NORMAL"
    allowed_to_continue = total_cost < hard and (total_cost < preferred or projected_total <= hard)
    pending = len(list(Path(args.records).glob("*/*/*/*.pending")))
    payload = {
        "schema": "trajuq_rq5_live_cost_ledger_v1",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": freeze_sha,
        "budget_authorization_sha256": sha256_bytes(Path(args.authorization).read_bytes()),
        "completed_requests": sum(completed.values()),
        "expected_requests": sum(expected.values()),
        "active_pending_requests": pending,
        "cumulative_cost_rmb": total_cost,
        "projected_completion_cost_rmb": projected_total,
        "preferred_target_rmb": preferred,
        "warning_threshold_rmb": warning,
        "absolute_hard_stop_rmb": hard,
        "cost_watch_mode": mode,
        "paid_execution_allowed": allowed_to_continue,
        "cells": cells,
    }
    atomic_json(Path(args.output), payload)
    print(json.dumps({key: payload[key] for key in (
        "completed_requests", "expected_requests", "active_pending_requests",
        "cumulative_cost_rmb", "projected_completion_cost_rmb", "cost_watch_mode",
        "paid_execution_allowed",
    )}, indent=2))


if __name__ == "__main__":
    main()
