#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def evaluate_gate(macro_payload: dict, primary_cfg: dict) -> dict:
    gate = primary_cfg["statistics"]["rq1_claim_gate"]
    primary = str(primary_cfg["statistics"]["primary_endpoint"])
    if primary != "hierarchical_macro_auroc_improvement":
        raise ValueError(f"Unsupported frozen primary endpoint: {primary}")

    auroc = macro_payload.get("macro", {}).get("auroc", {})
    ci_low = float(auroc["improvement_ci_low"])
    improvement = float(auroc["improvement_macro"])
    positive_families = int(auroc.get("positive_model_families", 0))
    n_families = int(auroc.get("n_model_families", 0))
    complete = macro_payload.get("cell_set_complete") is True

    ci_rule = bool(gate.get("macro_auroc_ci_low_gt_zero", True))
    min_families = int(gate.get("min_positive_model_families", 2))

    conditions = {
        "complete_frozen_cell_set": complete,
        "macro_auroc_improvement_positive": improvement > 0,
        "macro_auroc_ci_low_gt_zero": (ci_low > 0) if ci_rule else True,
        "positive_model_families_at_least_minimum": positive_families >= min_families,
        "three_primary_model_families_present": n_families == 3,
    }
    supported = all(conditions.values())

    return {
        "protocol": "trajuq_rq1_claim_gate_v1",
        "primary_endpoint": primary,
        "decision": (
            "SUPPORT_POSITIVE_RQ1_CLAIM"
            if supported
            else "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM"
        ),
        "claim_supported": supported,
        "frozen_rules": {
            "macro_auroc_ci_low_gt_zero": ci_rule,
            "min_positive_model_families": min_families,
        },
        "observed": {
            "macro_auroc_improvement": improvement,
            "macro_auroc_ci_low": ci_low,
            "macro_auroc_ci_high": float(auroc["improvement_ci_high"]),
            "positive_model_families": positive_families,
            "n_model_families": n_families,
        },
        "conditions": conditions,
        "interpretation": (
            "This gate controls wording of the positive RQ1 claim only. "
            "A negative gate result is a valid scientific outcome and must not "
            "trigger post-hoc test-set redesign."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the pre-registered RQ1 positive-claim gate."
    )
    parser.add_argument("--macro", required=True)
    parser.add_argument(
        "--primary-config",
        default="configs/experiments/primary.yaml",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    macro = json.loads(Path(args.macro).read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(args.primary_config).read_text(encoding="utf-8"))
    payload = evaluate_gate(macro, cfg)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
