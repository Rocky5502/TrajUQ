#!/usr/bin/env python
"""Render authorized RQ2/RQ3 paper macros from immutable aggregates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SHA256 = {
    "rq2": "eb12c0f5b6d7227bb13553aa3f7c5b262adf45c012002bc3b0a4ce10cf9b7268",
    "cap_025": "829b769319f0bdd7e6e5aff10b2829b94c857f695ee2b36fa7409ed7eb45bdbb",
    "cap_050": "b61c7975fd69be615ec6698ac9f0a4a5855f0d3dfda7735d523ee65e8ae3dea4",
    "cap_075": "68faa1ef48ec45ee343aa225efa1e799e47739b698bb09c2827a543f72ac9761",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def validate_rq2(value: dict[str, Any]) -> None:
    if value.get("protocol") != "trajuq_vs_repeated_macro_v2":
        raise ValueError("RQ2 protocol mismatch")
    if value.get("comparison") != "single_trajectory_trajuq_vs_five_sample_baseline":
        raise ValueError("RQ2 comparison mismatch")
    if not value.get("cell_set_complete"):
        raise ValueError("RQ2 cell set is incomplete")
    if value.get("expected_cells") != 12 or value.get("observed_cells") != 12:
        raise ValueError("RQ2 must contain exactly 12/12 comparison cells")
    required = {"auroc", "auprc", "aurc", "ece15", "nll"}
    if not required <= set(value.get("macro", {})):
        raise ValueError("RQ2 macro metrics are incomplete")


def validate_cap(value: dict[str, Any], key: str, fraction: float, primary: bool) -> None:
    if value.get("protocol") != "trajuq_adaptive_macro_v1":
        raise ValueError(f"{key} protocol mismatch")
    if value.get("execution_semantics") != "selective_second_pass_rerun":
        raise ValueError(f"{key} execution semantics mismatch")
    if value.get("fraction_key") != key or value.get("max_refine_fraction") != fraction:
        raise ValueError(f"{key} cap mismatch")
    if bool(value.get("is_primary_fraction")) is not primary:
        raise ValueError(f"{key} primary/sensitivity role mismatch")
    if not value.get("cell_set_complete"):
        raise ValueError(f"{key} cell set is incomplete")
    if value.get("expected_cells") != 18 or value.get("observed_cells") != 18:
        raise ValueError(f"{key} must contain exactly 18/18 cells")


def fixed(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def signed(value: float, digits: int = 4) -> str:
    return f"{float(value):+.{digits}f}"


def command(name: str, value: str) -> list[str]:
    return [f"\\providecommand{{\\{name}}}{{}}", f"\\renewcommand{{\\{name}}}{{{value}}}"]


def render(paths: dict[str, Path]) -> str:
    for key, path in paths.items():
        observed = sha256(path)
        if observed != EXPECTED_SHA256[key]:
            raise ValueError(f"immutable source hash mismatch for {key}: {observed}")

    rq2 = load(paths["rq2"])
    caps = {key: load(paths[key]) for key in ("cap_025", "cap_050", "cap_075")}
    validate_rq2(rq2)
    validate_cap(caps["cap_025"], "cap_025", 0.25, False)
    validate_cap(caps["cap_050"], "cap_050", 0.5, True)
    validate_cap(caps["cap_075"], "cap_075", 0.75, False)

    lines = [
        "% GENERATED FILE -- DO NOT TYPE RESULT VALUES BY HAND.",
        "% Authorized RQ2/RQ3 manuscript interface; RQ4/RQ5 are supplied by separate final interfaces.",
    ]
    source_names = {
        "rq2": "results/repeated_sampling/macro_trajuq_vs_repeated.json",
        "cap_025": "results/adaptive/macro_cap_025.json",
        "cap_050": "results/adaptive/macro_cap_050.json",
        "cap_075": "results/adaptive/macro_cap_075.json",
    }
    for key in paths:
        lines.append(f"% source {key}: {source_names[key]} sha256={EXPECTED_SHA256[key]}")

    for metric, suffix in {"auroc": "AUROC", "auprc": "AUPRC", "aurc": "AURC"}.items():
        node = rq2["macro"][metric]
        lines += command(f"Repeated{suffix}", fixed(node["baseline_macro"]))
        lines += command(f"RepeatedTraj{suffix}", fixed(node["candidate_macro"]))
        lines += command(f"RepeatedDelta{suffix}", signed(node["improvement_macro"]))
        lines += command(f"RepeatedDelta{suffix}Low", signed(node["improvement_ci_low"]))
        lines += command(f"RepeatedDelta{suffix}High", signed(node["improvement_ci_high"]))
    for metric, suffix in (("ece15", "ECE"), ("nll", "NLL")):
        node = rq2["macro"][metric]
        lines += command(f"RepeatedDelta{suffix}", signed(node["improvement_macro"]))
        lines += command(f"RepeatedDelta{suffix}Low", signed(node["improvement_ci_low"]))
        lines += command(f"RepeatedDelta{suffix}High", signed(node["improvement_ci_high"]))
    lines += command("RepeatedAuxWallTime", fixed(rq2["auxiliary_compute"]["mean_test_auxiliary_wall_time_seconds_per_example"], 2))
    lines += command("RepeatedCellCount", str(rq2["expected_cells"]))
    lines += command("RepeatedDatasetCount", str(len(rq2["eligible_datasets"])))

    primary = caps["cap_050"]
    macro = primary["macro"]
    intervals = primary["intervals"]
    values = {
        "AdaptiveBaseAccuracy": fixed(macro["base_accuracy"]),
        "AdaptiveBaseCost": "1.0000",
        "AdaptiveRefinedAccuracy": fixed(macro["refined_accuracy"]),
        "AdaptiveRefinedCost": fixed(macro["refined_cost_ratio_to_base"]),
        "AdaptiveRandomAccuracy": fixed(macro["random_expected_accuracy"]),
        "AdaptiveRandomCost": fixed(macro["nominal_cost_ratio_to_base"]),
        "AdaptiveAccuracy": fixed(macro["adaptive_accuracy"]),
        "AdaptiveCost": fixed(macro["nominal_cost_ratio_to_base"]),
        "AdaptiveCostToFull": fixed(macro["nominal_cost_ratio_to_refined"]),
        "AdaptiveSelectedFraction": fixed(macro["selected_fraction"]),
        "AdaptiveDeltaBase": signed(macro["delta_vs_base"]),
        "AdaptiveDeltaRandom": signed(macro["delta_vs_random"]),
        "AdaptiveDeltaRefined": signed(macro["delta_vs_refined"]),
    }
    for name, value in values.items():
        lines += command(name, value)
    for key, suffix in (("delta_vs_base", "Base"), ("delta_vs_random", "Random"), ("delta_vs_refined", "Refined")):
        lines += command(f"AdaptiveDelta{suffix}Low", signed(intervals[key]["ci_low"]))
        lines += command(f"AdaptiveDelta{suffix}High", signed(intervals[key]["ci_high"]))

    for key, prefix in (("cap_025", "AdaptiveSensitivityTwentyFive"), ("cap_075", "AdaptiveSensitivitySeventyFive")):
        node = caps[key]["macro"]
        lines += command(f"{prefix}Accuracy", fixed(node["adaptive_accuracy"]))
        lines += command(f"{prefix}Cost", fixed(node["nominal_cost_ratio_to_base"]))
        lines += command(f"{prefix}SelectedFraction", fixed(node["selected_fraction"]))

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", default="paper/generated_authorized_rq23.tex", type=Path)
    args = parser.parse_args()
    root = args.source_root.resolve()
    paths = {
        "rq2": root / "results/repeated_sampling/macro_trajuq_vs_repeated.json",
        "cap_025": root / "results/adaptive/macro_cap_025.json",
        "cap_050": root / "results/adaptive/macro_cap_050.json",
        "cap_075": root / "results/adaptive/macro_cap_075.json",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(paths), encoding="utf-8", newline="\n")
    print(json.dumps({"status": "PASS", "output": str(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
