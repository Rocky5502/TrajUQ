#!/usr/bin/env python3
"""Run the predeclared zero-GPU RQ2 K=2/3/5 efficiency sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


EXPECTED_PROTOCOL = "trajuq_rq2_sample_efficiency_exploratory_v1"
METRIC_SOURCES = {
    "auroc": "auroc",
    "auprc": "auprc",
    "aurc": "aurc",
    "brier": "brier",
    "nll": "nll",
    "ece15": "ece",
    "risk_at_50_coverage": "risk_at_50_coverage",
    "risk_at_80_coverage": "risk_at_80_coverage",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def option_values(argv: list[str], option: str) -> list[str]:
    return [argv[index + 1] for index, item in enumerate(argv[:-1]) if item == option]


def option_value(argv: list[str], option: str) -> str:
    values = option_values(argv, option)
    if len(values) != 1:
        raise ValueError(f"expected one {option}; found {len(values)}")
    return values[0]


def metric_payload(payload: dict[str, Any]) -> dict[str, float]:
    values = {
        output_name: float(payload[source_name])
        for output_name, source_name in METRIC_SOURCES.items()
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite RQ2 sensitivity metric")
    return values


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        candidate = Path(temporary)
        if candidate.exists():
            candidate.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/analysis/rq2_sample_efficiency.json"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default="python")
    args = parser.parse_args()

    root = args.root.resolve()
    contract_path = args.contract.resolve()
    contract = load_json(contract_path)
    if contract.get("protocol") != EXPECTED_PROTOCOL or contract.get("confirmatory") is not False:
        raise SystemExit("RQ2 efficiency contract is not the frozen exploratory protocol")
    if contract.get("sample_counts") != {
        "2": [2027, 2028],
        "3": [2027, 2028, 2029],
        "5": [2027, 2028, 2029, 2030, 2031],
    }:
        raise SystemExit("RQ2 efficiency seed subsets drifted")
    for binding in contract["source_bindings"].values():
        path = root / binding["path"]
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise SystemExit(f"RQ2 efficiency source drift: {binding['path']}")

    final_manifest = load_json(root / contract["source_bindings"]["final_manifest"]["path"])
    if final_manifest.get("status") != "FINAL" or final_manifest.get("audit_status") != "PASS":
        raise SystemExit("RQ2 final evidence is not authorized for sensitivity analysis")
    plan = load_json(root / contract["source_bindings"]["plan"]["path"])
    if plan.get("status") != "PASS" or len(plan.get("cells", [])) != 12:
        raise SystemExit("RQ2 plan is not the frozen 12-cell matrix")

    output_root = args.output_root.resolve()
    final_output = output_root / "rq2_sample_efficiency.json"
    if final_output.exists():
        raise SystemExit("Refusing to overwrite existing RQ2 sensitivity evidence")

    cell_results: dict[str, dict[str, Any]] = {}
    macro_values: dict[str, dict[str, list[float]]] = {
        key: {metric: [] for metric in METRIC_SOURCES}
        for key in contract["sample_counts"]
    }
    macro_cost: dict[str, list[float]] = {key: [] for key in contract["sample_counts"]}

    for cell in plan["cells"]:
        label = str(cell["cell"])
        evaluate = list(cell["evaluate_argv"])
        cal_samples = option_values(evaluate, "--cal-sample")
        test_samples = option_values(evaluate, "--test-sample")
        if len(cal_samples) != 5 or len(test_samples) != 5:
            raise SystemExit(f"{label}: frozen five-sample paths are incomplete")
        expected_seeds = [2027, 2028, 2029, 2030, 2031]
        for paths in (cal_samples, test_samples):
            observed = [int(Path(path).name.split("seed-")[1].split(".")[0]) for path in paths]
            if observed != expected_seeds:
                raise SystemExit(f"{label}: frozen sample order differs from nested seed rule")

        cell_results[label] = {}
        for k_text, seeds in contract["sample_counts"].items():
            k = int(k_text)
            cell_dir = output_root / "cells" / label.replace(":", "__")
            metrics_path = cell_dir / f"k{k}.metrics.json"
            if metrics_path.exists():
                raise SystemExit(f"Refusing to overwrite {metrics_path}")
            command = [
                args.python,
                str(root / "scripts/evaluate_repeated_sampling.py"),
                "--cal-target",
                option_value(evaluate, "--cal-target"),
            ]
            for path in cal_samples[:k]:
                command.extend(["--cal-sample", path])
            command.extend(["--test-target", option_value(evaluate, "--test-target")])
            for path in test_samples[:k]:
                command.extend(["--test-sample", path])
            command.extend(
                [
                    "--expected-samples",
                    str(k),
                    "--seed",
                    "2027",
                    "--output",
                    str(metrics_path),
                ]
            )
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, cwd=root, check=True, stdout=subprocess.DEVNULL)
            metrics = load_json(metrics_path)
            if metrics.get("sample_count") != k or metrics.get("sample_seeds") != seeds:
                raise SystemExit(f"{label}/K={k}: evaluated seed set drifted")
            values = metric_payload(metrics)
            cost = float(
                metrics["auxiliary_compute"]["test"][
                    "mean_auxiliary_wall_time_seconds_per_example"
                ]
            )
            if not math.isfinite(cost) or cost < 0:
                raise SystemExit(f"{label}/K={k}: invalid compute accounting")
            cell_results[label][k_text] = {
                "metrics": values,
                "mean_auxiliary_wall_time_seconds_per_example": cost,
                "source": str(metrics_path.relative_to(output_root)),
                "sha256": sha256_file(metrics_path),
            }
            for metric, value in values.items():
                macro_values[k_text][metric].append(value)
            macro_cost[k_text].append(cost)

            if k == 5:
                authorized = load_json(root / str(cell["repeated_metrics"]))
                authorized_values = metric_payload(authorized)
                if any(
                    abs(values[name] - authorized_values[name]) > 1e-12
                    for name in METRIC_SOURCES
                ):
                    raise SystemExit(f"{label}: K=5 does not reproduce authorized metrics")

    macro = {
        k: {
            "metrics": {
                name: sum(values) / len(values)
                for name, values in metrics.items()
            },
            "mean_auxiliary_wall_time_seconds_per_example": (
                sum(macro_cost[k]) / len(macro_cost[k])
            ),
            "n_cells": len(macro_cost[k]),
        }
        for k, metrics in macro_values.items()
    }
    k5_cost = macro["5"]["mean_auxiliary_wall_time_seconds_per_example"]
    k5_metrics = macro["5"]["metrics"]
    for k in contract["sample_counts"]:
        cost = macro[k]["mean_auxiliary_wall_time_seconds_per_example"]
        macro[k]["cost_fraction_of_k5"] = cost / k5_cost
        macro[k]["saved_auxiliary_seconds_per_example_vs_k5"] = k5_cost - cost
        macro[k]["metric_delta_vs_k5"] = {
            name: macro[k]["metrics"][name] - k5_metrics[name]
            for name in METRIC_SOURCES
        }
    payload = {
        "schema": "trajuq_rq2_sample_efficiency_results_v1",
        "status": "PASS",
        "classification": "secondary_exploratory",
        "new_model_generations": 0,
        "contract": {
            "path": (
                str(contract_path.relative_to(root))
                if contract_path.is_relative_to(root)
                else contract_path.name
            ),
            "sha256": sha256_file(contract_path),
        },
        "source_bindings": contract["source_bindings"],
        "sample_counts": contract["sample_counts"],
        "aggregation": contract["aggregation"],
        "macro": macro,
        "cells": cell_results,
        "interpretation_guard": (
            "Cost/evidence sensitivity only; does not replace the authorized K=5 result "
            "and supports no new confirmatory claim."
        ),
    }
    atomic_json(final_output, payload)
    print(json.dumps({"status": "PASS", "output": str(final_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
