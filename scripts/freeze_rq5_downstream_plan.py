#!/usr/bin/env python
"""Freeze the RQ5 post-data/pre-outcome-analysis orchestration plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/rq5/rq5_downstream_plan.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    census_path = root / "results/rq5/census/rq5_evidence_census.json"
    census = json.loads(census_path.read_text(encoding="utf-8"))
    if census.get("aggregate_outcomes_computed") is not False:
        raise SystemExit("RQ5 outcome-blind chronology cannot be established")
    source_paths = [
        "scripts/rq5_downstream_stage.py",
        "scripts/freeze_rq5_downstream_plan.py",
        "src/trajuq/scoring.py",
        "src/trajuq/repeated_sampling.py",
        "src/trajuq/evaluation.py",
        "src/trajuq/statistics.py",
        "src/trajuq/calibration.py",
        "configs/experiments/tmlr_rq5_blackbox.yaml",
    ]
    input_paths = [
        "results/rq5/TMLR_RQ5_BLACKBOX_FREEZE.json",
        "results/rq5/validated/response_registry.jsonl",
        "results/rq5/validated/registry_manifest.json",
        "results/rq5/validated/registry_manifest.recovered.json",
        "results/rq5/reconciliation/rq5_terminal_reconciliation.json",
        "results/rq5/reconciliation/rq5_terminal_ledger.jsonl",
        "results/rq5/census/rq5_evidence_census.json",
    ]
    plan = {
        "schema": "trajuq_rq5_downstream_plan_v1",
        "status": "PASS",
        "chronology": "POST_DATA_PRE_OUTCOME_ANALYSIS",
        "timing_note": "Frozen after API data collection and terminal reconciliation, before aggregate RQ5 outcome analysis or inspection.",
        "claim_authorized": False,
        "network_calls_allowed": False,
        "inputs": {
            "response_registry": "results/rq5/validated/response_registry.jsonl",
            "terminal_reconciliation": "results/rq5/reconciliation/rq5_terminal_reconciliation.json",
            "terminal_ledger": "results/rq5/reconciliation/rq5_terminal_ledger.jsonl",
        },
        "response_registry_sha256": sha256(root / "results/rq5/validated/response_registry.jsonl"),
        "terminal_ledger_sha256": sha256(root / "results/rq5/reconciliation/rq5_terminal_ledger.jsonl"),
        "source_bindings": [
            {"path": path, "sha256": sha256(root / path)} for path in source_paths
        ],
        "input_bindings": [
            {"path": path, "sha256": sha256(root / path)} for path in input_paths
        ],
        "eligibility": {
            "primary": "terminal category valid; group has at least two valid registered repeats",
            "unit": "model x dataset x split x example_id repeated-sample group",
            "group_prediction": "modal task-canonical answer; ties resolved by lowest registered repeat index",
            "group_error_label": "frozen scorer correctness of deterministic modal-answer representative",
            "complete_five_groups": "SECONDARY_EXPLORATORY sensitivity only",
            "malformed_provider_failure_locked": "retained in accounting; never replayed; excluded where no valid response value exists",
        },
        "features": {
            "primary_behavioral": [
                "sample_disagreement",
                "sample_vote_entropy",
                "sample_unique_fraction",
                "verbalized_risk",
                "confidence_std",
            ],
            "combined": "standardized logistic regression fitted independently per model-dataset calibration cell",
            "semantic_response_consistency": "NOT_FEASIBLE_WITHOUT_NEW_UNREGISTERED_SEMANTIC_MODEL",
            "provider_reasoning_metadata": "diagnostic only; not hidden-state or mechanistic evidence",
        },
        "registered_science": {
            "calibration_fit": "calibration_partition_only",
            "test_evaluation": "heldout_test_only",
            "equal_cell_weight": True,
            "no_pooling_across_cells": True,
            "primary_endpoint": "equal-cell macro AURC improvement of behavioral_combined over verbalized_confidence",
            "metrics": ["aurc", "auroc", "auprc", "brier", "nll", "ece_15", "risk_at_50_coverage", "risk_at_80_coverage"],
            "bootstrap": "paired_class_stratified_within_cell_equal_cell_macro",
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 2027,
            "confidence_level": 0.95,
            "confirmatory_family": [
                "rq5_behavioral_selective_prediction",
                "rq5_cross_model_heterogeneity",
                "rq5_observability_gap",
            ],
            "multiplicity": "Holm-Bonferroni over estimable confirmatory p-values; non-estimable hypotheses retained explicitly",
            "alpha": 0.05,
        },
        "hypothesis_operationalization": {
            "rq5_behavioral_selective_prediction": "positive AURC improvement of combined behavioral signal over verbalized-confidence risk",
            "rq5_cross_model_heterogeneity": "model-level range in the primary AURC improvement; descriptive bootstrap only because no valid zero-null calibration was prospectively specified",
            "rq5_observability_gap": "NOT_IDENTIFIABLE if model identity is unmatched across observability regimes; descriptive synthesis cannot authorize a causal observability claim",
        },
        "secondary": [
            "complete-five-sample groups",
            "leave-one-model-out",
            "leave-one-dataset-out",
            "per-cell missingness characterization",
        ],
        "stages": [
            "validation",
            "feature_extraction",
            "calibration_fit",
            "heldout_evaluation",
            "macro_aggregate",
            "paired_bootstrap",
            "multiplicity_correction",
            "hypothesis_gate",
            "missingness_sensitivity",
            "finalize",
        ],
        "firewall": [
            "No API calls or request replay.",
            "No test labels used in calibration fitting.",
            "No heterogeneous row pooling across cells.",
            "All positive, negative, null, heterogeneous, malformed, and missingness results are retained.",
            "The corrupted original registry manifest remains unchanged and separately hash-bound.",
        ],
    }
    payload = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output = args.output if args.output.is_absolute() else root / args.output
    write_new(output, payload)
    write_new(Path(str(output) + ".sha256"), (hashlib.sha256(payload).hexdigest() + "\n").encode())


if __name__ == "__main__":
    main()
