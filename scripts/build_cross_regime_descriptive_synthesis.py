#!/usr/bin/env python3
"""Build the final, non-pooling cross-regime descriptive synthesis.

This builder deliberately keeps each regime's registered estimand and cost
ledger separate.  It is a manuscript interface, not a new scientific claim
gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/analysis/cross_regime_descriptive_synthesis_v2.json"
OUT = ROOT / "results/generated/secondary/cross_regime_synthesis"
TEX = ROOT / "paper/generated_synthesis.tex"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def f4(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    config = load(CONFIG)
    for binding in config["source_bindings"]:
        path = ROOT / binding["path"]
        actual = sha256(path)
        if actual != binding["sha256"]:
            raise SystemExit(f"source drift: {binding['path']} expected {binding['sha256']} got {actual}")

    primary = load(ROOT / "results/primary_analysis/macro_primary.json")
    adaptive = load(ROOT / "results/adaptive/macro_cap_050.json")
    repeated = load(ROOT / "results/repeated_sampling/macro_trajuq_vs_repeated.json")
    rq2_cost = load(ROOT / "results/generated/secondary/rq2_sample_efficiency/rq2_cost_utility.json")
    rq4 = load(ROOT / "results/generated/final/rq4_scientific_claim_registry.json")
    rq5 = load(ROOT / "results/generated/final/rq5_scientific_claim_registry.json")
    rq5_hetero = load(ROOT / "results/generated/secondary/rq5_heterogeneity/heterogeneity_summary.json")

    early_path = ROOT / config["optional_secondary_source"]["path"]
    early_manifest = load(early_path) if early_path.exists() else None
    early = None
    if early_manifest is not None:
        if early_manifest.get("status") != "PASS" or early_manifest.get("scientific_claim_authority") is not False:
            raise SystemExit("optional RQ4 early-actionability manifest is not a PASS secondary artifact")
        sidecar = early_path.with_name("manifest.json.sha256").read_text(encoding="ascii").strip()
        if sha256(early_path) != sidecar:
            raise SystemExit("optional RQ4 early-actionability manifest sidecar drift")
        for artifact in early_manifest["artifacts"]:
            artifact_path = early_path.parent / artifact["path"]
            if sha256(artifact_path) != artifact["sha256"] or artifact_path.stat().st_size != artifact["bytes"]:
                raise SystemExit(f"optional RQ4 early-actionability artifact drift: {artifact['path']}")
        early = load(early_path.parent / "early_actionability.json")
        if early.get("status") != "PASS" or early.get("scientific_claim_authority") is not False:
            raise SystemExit("optional RQ4 early-actionability result is not a PASS secondary artifact")
    early_effects = {row["fraction"]: row for row in early["trajectory_vs_endpoint"]} if early else {}

    rows = [
        {
            "regime": "Native diffusion trajectory",
            "observed_information": "Native denoising trajectory",
            "registered_comparison": "TrajUQ vs terminal-entropy baseline",
            "principal_result": (
                f"Delta AUROC {f4(primary['macro']['auroc']['improvement_macro'])} "
                f"[{f4(primary['macro']['auroc']['improvement_ci_low'])}, "
                f"{f4(primary['macro']['auroc']['improvement_ci_high'])}]; "
                f"Delta AURC {f4(primary['macro']['aurc']['improvement_macro'])} "
                f"[{f4(primary['macro']['aurc']['improvement_ci_low'])}, "
                f"{f4(primary['macro']['aurc']['improvement_ci_high'])}]"
            ),
            "actionability": (
                f"Selective refinement at cap 0.50: Delta accuracy vs random "
                f"{f4(adaptive['intervals']['delta_vs_random']['point'])} "
                f"[{f4(adaptive['intervals']['delta_vs_random']['ci_low'])}, "
                f"{f4(adaptive['intervals']['delta_vs_random']['ci_high'])}]"
            ),
            "cost": f"{adaptive['macro']['nominal_cost_ratio_to_base']:.3f}x base compute",
            "boundary": "Within registered diffusion cells only",
        },
        {
            "regime": "Open autoregressive trace",
            "observed_information": "Token-level uncertainty trace",
            "registered_comparison": "Trajectory calibrator vs mean token entropy",
            "principal_result": (
                f"Delta AURC {f4(rq4['confirmatory_claims']['h4a']['estimate'])} "
                f"[{f4(rq4['confirmatory_claims']['h4a']['ci_95'][0])}, "
                f"{f4(rq4['confirmatory_claims']['h4a']['ci_95'][1])}]"
            ),
            "actionability": (
                "Fixed-prefix analysis is secondary exploratory"
                if early is None
                else (
                    f"Secondary prefix Delta AURC is positive at all fixed fractions: "
                    f"{f4(early_effects[0.25]['estimate'])} at 25% and {f4(early_effects[1.0]['estimate'])} at 100%"
                )
            ),
            "cost": "One generation for trace; five for semantic disagreement comparator",
            "boundary": "One pinned open-weight model; no diffusion-to-AR transfer claim",
        },
        {
            "regime": "Repeated sampling",
            "observed_information": "Cross-generation semantic disagreement",
            "registered_comparison": "Single-trajectory TrajUQ vs five samples",
            "principal_result": (
                f"Delta AURC {f4(repeated['macro']['aurc']['improvement_macro'])} "
                f"[{f4(repeated['macro']['aurc']['improvement_ci_low'])}, "
                f"{f4(repeated['macro']['aurc']['improvement_ci_high'])}]"
            ),
            "actionability": "K=2/3/5 improves ranking and selective risk; ECE15 worsens",
            "cost": f"K=2/3/5: {rq2_cost['macro'][0]['seconds']:.1f}/{rq2_cost['macro'][1]['seconds']:.1f}/{rq2_cost['macro'][2]['seconds']:.1f} auxiliary seconds/example",
            "boundary": "Three-point sensitivity; no post-hoc optimal K",
        },
        {
            "regime": "Hosted black-box behavior",
            "observed_information": "Verbalized confidence plus repeated behavior",
            "registered_comparison": "Combined behavioral vs verbalized confidence",
            "principal_result": (
                f"Delta AURC {f4(rq5['decisions']['rq5_behavioral_selective_prediction']['estimate'])} "
                f"[{f4(rq5['decisions']['rq5_behavioral_selective_prediction']['ci_95'][0])}, "
                f"{f4(rq5['decisions']['rq5_behavioral_selective_prediction']['ci_95'][1])}]; NOT SUPPORTED"
            ),
            "actionability": f"Descriptive heterogeneity: {rq5_hetero['cells_favoring_combined']}/18 cells favor combined",
            "cost": "Five provider responses per item",
            "boundary": "Unmatched models; cross-observability gap not identifiable",
        },
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "regime_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema": "trajuq_cross_regime_descriptive_synthesis_v2",
        "status": "PASS",
        "classification": "DESCRIPTIVE_SYNTHESIS",
        "scientific_claim_authority": False,
        "config_sha256": sha256(CONFIG),
        "rows": rows,
        "optional_rq4_early_actionability_included": early is not None,
        "optional_rq4_early_actionability_manifest_sha256": sha256(early_path) if early is not None else None,
        "interpretation": {
            "supported": "Uncertainty utility depends on what is observable and on the inference cost used to obtain it.",
            "prohibited": "No causal observability law, raw cross-protocol model ranking, or pooled cross-regime effect.",
        },
    }
    summary_path = OUT / "summary.json"
    write_json_new(summary_path, summary)

    tex = rf"""% AUTO-GENERATED by scripts/build_cross_regime_descriptive_synthesis.py; do not edit.
% Classification: DESCRIPTIVE_SYNTHESIS; scientific_claim_authority=false.
\newcommand{{\CrossModelSynthesisState}}{{FINAL DESCRIPTIVE}}
\newcommand{{\CrossModelSynthesisTable}}{{%
\begin{{table}}[H]
\centering
\small
\setlength{{\tabcolsep}}{{2.2pt}}
\begin{{tabular}}{{@{{}}>{{\raggedright\arraybackslash}}p{{1.7cm}}>{{\raggedright\arraybackslash}}p{{2.5cm}}>{{\raggedright\arraybackslash}}p{{4.0cm}}>{{\raggedright\arraybackslash}}p{{3.5cm}}>{{\raggedright\arraybackslash}}p{{3.0cm}}@{{}}}}
\toprule
Regime & Observable evidence & Main within-regime result & Additional inference cost & Scientific boundary \\
\midrule
Native diffusion & Denoising path & $\Delta$AUROC $+{f4(primary['macro']['auroc']['improvement_macro'])}$ [{f4(primary['macro']['auroc']['improvement_ci_low'])}, {f4(primary['macro']['auroc']['improvement_ci_high'])}]; $\Delta$AURC $+{f4(primary['macro']['aurc']['improvement_macro'])}$ [{f4(primary['macro']['aurc']['improvement_ci_low'])}, {f4(primary['macro']['aurc']['improvement_ci_high'])}] & Risk-guided refinement beats matched random by {f4(adaptive['intervals']['delta_vs_random']['point'])} at {adaptive['macro']['nominal_cost_ratio_to_base']:.3f}$\times$ base compute & Matched diffusion cells only \\
Open AR & Token distributions over one trace & Trace vs mean entropy $\Delta$AURC $+{f4(rq4['confirmatory_claims']['h4a']['estimate'])}$ [{f4(rq4['confirmatory_claims']['h4a']['ci_95'][0])}, {f4(rq4['confirmatory_claims']['h4a']['ci_95'][1])}] & One trace; five-sample comparator requires separate generations & One pinned model; no transfer claim \\
Repeated sampling & Answer disagreement & Single trajectory vs five samples $\Delta$AURC $+{f4(repeated['macro']['aurc']['improvement_macro'])}$ [{f4(repeated['macro']['aurc']['improvement_ci_low'])}, {f4(repeated['macro']['aurc']['improvement_ci_high'])}] & $K=2/3/5$ costs {rq2_cost['macro'][0]['seconds']:.1f}/{rq2_cost['macro'][1]['seconds']:.1f}/{rq2_cost['macro'][2]['seconds']:.1f} s/example & No optimal $K$ selected \\
Hosted black box & Text, confidence, repeated behavior & Combined vs confidence $\Delta$AURC ${f4(rq5['decisions']['rq5_behavioral_selective_prediction']['estimate'])}$ [{f4(rq5['decisions']['rq5_behavioral_selective_prediction']['ci_95'][0])}, {f4(rq5['decisions']['rq5_behavioral_selective_prediction']['ci_95'][1])}]; not supported & Five responses per item & Unmatched models; effect not identifiable \\
\bottomrule
\end{{tabular}}
\caption{{Observability- and cost-stratified synthesis. Every number retains its own within-regime baseline and aggregation rule. Rows are not pooled or numerically ranked: the systems are unmatched, so the table does not identify a causal effect of observability.}}
\label{{tab:cross-model-synthesis}}
\end{{table}}}}

\newcommand{{\CrossModelAppendixTable}}{{%
\begin{{table*}}[p]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{2.5pt}}
\begin{{tabular}}{{p{{2.7cm}}p{{3.8cm}}p{{3.8cm}}p{{3.8cm}}}}
\toprule
Comparison & Comparable within & Not comparable across & Claim boundary \\
\midrule
Predictive/selective metrics & Same task, split, estimator, and aggregation & Unlike models, access interfaces, and protocols & No pooled cross-regime effect \\
Adaptive computation & Frozen diffusion refinement policy & AR prefixes or provider requests & Separate cost ledgers \\
Repeated sampling & K=2/3/5 on the same 12 cells & Native trace and hosted black-box endpoints & No post-hoc winner \\
Black-box behavior & Combined vs confidence on matched requests & Open-weight AR or diffusion systems & Gap not identifiable \\
\bottomrule
\end{{tabular}}
\caption{{Compatibility firewall for the descriptive synthesis. An apparent numerical alignment across rows does not create a matched estimand.}}
\label{{tab:cross-model-synthesis-compatibility}}
\end{{table*}}}}

\newcommand{{\CrossModelSignalTable}}{{%
\begin{{table*}}[p]
\centering
\footnotesize
\setlength{{\tabcolsep}}{{2pt}}
\begin{{tabular}}{{p{{2.4cm}}ccccp{{4.8cm}}}}
\toprule
Regime & Native path & Token trace & Repeats & Confidence & Cost unit \\
\midrule
Native diffusion & yes & no & optional & no & Base/refinement compute ratio \\
Open autoregressive & no & yes & optional & no & Generation count \\
Repeated sampling & no & no & yes & no & Auxiliary seconds and K \\
Hosted black box & no & no & yes & yes & Provider request count \\
\bottomrule
\end{{tabular}}
\caption{{Structural signal availability and regime-specific cost accounting. Unlike cost units are intentionally not collapsed into a single efficiency score.}}
\label{{tab:cross-model-signal-availability}}
\end{{table*}}}}
"""
    TEX.write_text(tex, encoding="utf-8")

    manifest = {
        "schema": "trajuq_cross_regime_descriptive_synthesis_manifest_v2",
        "status": "PASS",
        "scientific_claim_authority": False,
        "config": {"path": str(CONFIG.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(CONFIG)},
        "artifacts": [
            {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(path)}
            for path in (summary_path, csv_path, TEX)
        ],
    }
    manifest_path = OUT / "manifest.json"
    write_json_new(manifest_path, manifest)
    (OUT / "manifest.json.sha256").write_text(sha256(manifest_path) + "\n", encoding="ascii")
    print(json.dumps({"status": "PASS", "manifest_sha256": sha256(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
