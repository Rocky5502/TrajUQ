#!/usr/bin/env python3
"""Render the authorized RQ4 hypothesis registry as a LaTeX interface.

The renderer is fail-closed: it accepts only a PASS registry with scientific
claim authorization and all three registered hypotheses present. Numerical
values are read from the immutable scientific registry, never embedded here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "h4a": "Within-regime trace vs. endpoint",
    "h4b": "Diffusion dynamics beyond shared features",
    "h4c": "Within-regime vs. cross-paradigm transfer",
}


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render(registry_path: Path, output_path: Path) -> None:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("status") != "PASS" or registry.get("claim_authorized") is not True:
        raise SystemExit("RQ4 scientific registry is not claim-authorized PASS")
    claims = registry.get("confirmatory_claims", {})
    if set(claims) != set(LABELS):
        raise SystemExit("RQ4 registry must contain exactly H4a, H4b, and H4c")

    rows: list[str] = []
    for key in ("h4a", "h4b", "h4c"):
        claim = claims[key]
        low, high = claim["ci_95"]
        decision = str(claim["decision"])
        if decision != "SUPPORTED":
            raise SystemExit(f"Unexpected RQ4 decision for {key}: {decision}")
        rows.append(
            f"H4{key[-1]} & {LABELS[key]} & "
            f"{_fmt(float(claim['estimate']))} "
            f"[{_fmt(float(low))}, {_fmt(float(high))}] & "
            f"{float(claim['holm_adjusted_p']):.4f} & Gate met \\\\"
        )

    source_hash = next(
        binding["sha256"]
        for binding in registry["bindings"]
        if binding["role"] == "hypothesis_gate"
    )
    content = "\n".join(
        [
            "% Generated from results/generated/final/rq4_scientific_claim_registry.json.",
            f"% Bound hypothesis-gate SHA-256: {source_hash}",
            r"\newcommand{\RQFourRegistryTable}{%",
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{3.2pt}",
            r"\begin{tabular}{@{}c>{\raggedright\arraybackslash}p{4.6cm}>{\centering\arraybackslash}p{3.1cm}cc@{}}",
            r"\toprule",
            r"Gate & Registered contrast & Estimate [95\% CI] & Holm $p$ & Decision \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{RQ4 confirmatory gates. Intervals are paired class-stratified 95\% CIs; positive estimates meet the corresponding registered directional test. H4b isolates diffusion-specific dynamics beyond shared distributional features, and H4c compares within-regime calibration with cross-paradigm transfer; neither is a causal observability contrast.}",
            r"\label{tab:rq4-confirmatory-gates}",
            r"\end{table}%",
            r"}",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.registry, args.output)


if __name__ == "__main__":
    main()
