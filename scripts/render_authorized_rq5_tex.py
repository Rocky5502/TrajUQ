#!/usr/bin/env python
"""Render the authorized RQ5 manuscript interface from claim-bound artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def main() -> None:
    claim_path = ROOT / "results/generated/final/rq5_scientific_claim_registry.json"
    manifest_path = ROOT / "results/generated/final/rq5_manifest.json"
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if claim.get("status") != "PASS" or claim.get("claim_authorized") is not True:
        raise SystemExit("RQ5 scientific claim registry is not authorized")
    if manifest.get("status") != "FINAL" or manifest.get("claim_authorized") is not True:
        raise SystemExit("RQ5 final artifact manifest is not authorized")
    for binding in claim["bindings"]:
        path = ROOT / binding["path"]
        if sha256(path) != binding["sha256"]:
            raise SystemExit(f"RQ5 manuscript binding drift: {binding['path']}")

    validation = json.loads((ROOT / "results/rq5/analysis/validation_summary.json").read_text(encoding="utf-8"))
    macro = json.loads((ROOT / "results/rq5/analysis/macro_aggregate.json").read_text(encoding="utf-8"))["macro"]
    decision = claim["decisions"]["rq5_behavioral_selective_prediction"]
    heterogeneity = claim["decisions"]["rq5_cross_model_heterogeneity"]
    ci_low, ci_high = decision["ci_95"]
    rows = []
    labels = {
        "behavioral_combined": "Combined behavioral",
        "repeated_disagreement": "Repeated-answer disagreement",
        "verbalized_confidence": "Verbalized confidence",
    }
    for signal in ("behavioral_combined", "repeated_disagreement", "verbalized_confidence"):
        value = macro[signal]
        rows.append(
            f"{labels[signal]} & {fmt(value['aurc'])} & {fmt(value['auroc'])} & "
            f"{fmt(value['auprc'])} & {fmt(value['brier'])} & {fmt(value['ece_15'])} \\\\"
        )
    block = (
        "% AUTO-GENERATED from the authorized RQ5 scientific claim registry; do not edit.\n"
        f"\\newcommand{{\\RQFiveRequestedResponses}}{{{validation['planned_requests']}}}\n"
        f"\\newcommand{{\\RQFiveSavedResponses}}{{{validation['saved_responses']}}}\n"
        f"\\newcommand{{\\RQFiveValidatedResponses}}{{{validation['schema_valid_requests']}}}\n"
        f"\\newcommand{{\\RQFiveMalformedResponses}}{{{validation['malformed']}}}\n"
        "\\newcommand{\\RQFiveGeneratedResults}{%\n"
        "Across 18 equal-weight model--dataset cells, the registered combined behavioral signal did not improve the primary selective-risk endpoint over verbalized-confidence risk: "
        f"the AURC improvement was {signed(decision['estimate'])} "
        f"(95\\% CI [{signed(ci_low)}, {signed(ci_high)}]; Holm-adjusted $p={fmt(decision['holm_adjusted_p'])}$; {decision['decision'].replace('_', ' ').lower()}). "
        "The equal-cell macro AURC was "
        f"{fmt(macro['behavioral_combined']['aurc'])} for the combined signal and {fmt(macro['verbalized_confidence']['aurc'])} for verbalized confidence. "
        "Model-specific improvements were heterogeneous in sign "
        f"(descriptive range {fmt(heterogeneity['estimate'])}, bootstrap interval [{fmt(heterogeneity['ci_low'])}, {fmt(heterogeneity['ci_high'])}]), "
        "but this heterogeneity view has no confirmatory null calibration. The cross-observability gap remains non-identifiable because model identity is unmatched across regimes.\n\n"
        f"The attrition flow is {validation['planned_requests']:,} requested/accounted $\\rightarrow$ {validation['saved_responses']:,} saved $\\rightarrow$ {validation['schema_valid_requests']:,} schema-valid; "
        f"{validation['malformed']:,} malformed responses, {validation['provider_failure']} provider failure, and {validation['ambiguous_locked_no_replay']} ambiguous locked requests are retained in the accounting and were not replayed.\n\n"
        "\\begin{table}[t]\\centering\\small\n"
        "\\begin{tabular}{lrrrrr}\\toprule\n"
        "Signal & AURC$\\downarrow$ & AUROC$\\uparrow$ & AUPRC$\\uparrow$ & Brier$\\downarrow$ & ECE15$\\downarrow$ \\\\ \\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\\end{tabular}\n"
        "\\caption{RQ5 equal-cell held-out macro metrics. The confirmatory endpoint is the paired AURC improvement of the combined behavioral signal over verbalized confidence; the remaining columns are retained descriptive secondary metrics.}\n"
        "\\label{tab:rq5-authorized-macro}\\end{table}%\n"
        "}\n"
    ).encode("utf-8")
    generated_path = ROOT / "paper/generated_rq5.tex"
    interface_path = ROOT / "paper/generated/interfaces/rq5_macros.tex"
    atomic(generated_path, block)
    # Keep a single canonical copy of every LaTeX label.  The compatibility
    # interface delegates to the manuscript-level generated file instead of
    # duplicating the entire block (which makes source-only label audits fail
    # even when the interface is not included by main.tex).
    interface_block = (
        "% AUTO-GENERATED compatibility interface; do not edit.\n"
        "% Canonical RQ5 macros and labels live in paper/generated_rq5.tex.\n"
        "\\input{generated_rq5}\n"
    ).encode("utf-8")
    atomic(interface_path, interface_block)
    interface_manifest = {
        "schema": "trajuq_rq5_manuscript_interface_v1",
        "status": "PASS",
        "claim_registry_sha256": sha256(claim_path),
        "rq5_manifest_sha256": sha256(manifest_path),
        "generated_rq5_tex_sha256": sha256(generated_path),
        "rq5_macros_tex_sha256": sha256(interface_path),
    }
    atomic(
        ROOT / "paper/generated/interfaces/rq5_interface_manifest.json",
        (json.dumps(interface_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


if __name__ == "__main__":
    main()
