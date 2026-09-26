#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: str) -> dict:
    source = Path(path)
    if not source.exists():
        raise SystemExit(f"Missing result artifact: {source}")
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON artifact {source}: {exc}") from exc


def metric_pair(primary: dict, metric: str) -> tuple[float, float]:
    node = primary.get("macro", {}).get(metric)
    if not isinstance(node, dict):
        raise ValueError(f"Primary macro missing metric={metric}")
    return float(node["baseline_macro"]), float(node["candidate_macro"])


def require_primary(primary: dict) -> None:
    if primary.get("cell_set_complete") is not True:
        raise ValueError("Primary macro does not contain the complete frozen cell set.")
    if int(primary.get("expected_cell_count", -1)) != 18:
        raise ValueError("Primary macro expected cell count is not 18.")
    if int(primary.get("observed_cell_count", -1)) != 18:
        raise ValueError("Primary macro observed cell count is not 18.")
    for metric in ("auroc", "auprc", "brier", "ece15", "aurc"):
        metric_pair(primary, metric)


def require_claim_gate(claim: dict, primary: dict) -> None:
    decision = str(claim.get("decision", ""))
    if decision not in {
        "SUPPORT_POSITIVE_RQ1_CLAIM",
        "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM",
    }:
        raise ValueError("RQ1 claim gate has no valid frozen decision.")
    observed = claim.get("observed", {})
    primary_auroc = primary["macro"]["auroc"]
    checks = [
        (
            float(observed["macro_auroc_improvement"]),
            float(primary_auroc["improvement_macro"]),
        ),
        (
            float(observed["macro_auroc_ci_low"]),
            float(primary_auroc["improvement_ci_low"]),
        ),
        (
            float(observed["macro_auroc_ci_high"]),
            float(primary_auroc["improvement_ci_high"]),
        ),
    ]
    if any(abs(left - right) > 1e-12 for left, right in checks):
        raise ValueError("RQ1 claim gate does not match the supplied primary macro.")


def require_repeated(repeated: dict) -> None:
    if repeated.get("cell_set_complete") is not True:
        raise ValueError("Repeated-sampling macro cell set is incomplete.")
    if int(repeated.get("expected_cells", -1)) != 15:
        raise ValueError("Repeated-sampling macro expected cell count is not 15.")
    if int(repeated.get("observed_cells", -1)) != 15:
        raise ValueError("Repeated-sampling macro observed cell count is not 15.")
    for metric in ("auroc", "auprc", "aurc"):
        node = repeated.get("macro", {}).get(metric)
        if not isinstance(node, dict):
            raise ValueError(f"Repeated macro missing metric={metric}")
        float(node["baseline_macro"])
        float(node["candidate_macro"])


def require_adaptive(adaptive: dict) -> None:
    if adaptive.get("cell_set_complete") is not True:
        raise ValueError("Adaptive macro cell set is incomplete.")
    if int(adaptive.get("expected_cells", -1)) != 18:
        raise ValueError("Adaptive macro expected cell count is not 18.")
    if int(adaptive.get("observed_cells", -1)) != 18:
        raise ValueError("Adaptive macro observed cell count is not 18.")
    if adaptive.get("is_primary_fraction") is not True:
        raise ValueError("Adaptive artifact is not marked as the primary cap.")
    if abs(float(adaptive.get("max_refine_fraction", -1)) - 0.50) > 1e-12:
        raise ValueError("Adaptive primary macro is not the frozen 50% cap.")

    macro = adaptive.get("macro", {})
    for key in (
        "adaptive_accuracy",
        "base_accuracy",
        "refined_accuracy",
        "random_expected_accuracy",
        "delta_vs_random",
        "nominal_cost_ratio_to_base",
        "refined_cost_ratio_to_base",
    ):
        float(macro[key])


def fmt(value: float) -> str:
    return f"{float(value):.3f}"


def fmt_signed(value: float) -> str:
    return f"{float(value):+.3f}"


def build_macros(
    primary: dict,
    claim: dict,
    repeated: dict,
    adaptive: dict,
) -> dict[str, str]:
    require_primary(primary)
    require_claim_gate(claim, primary)
    require_repeated(repeated)
    require_adaptive(adaptive)

    macros: dict[str, str] = {}
    primary_names = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "brier": "Brier",
        "ece15": "ECE",
        "aurc": "AURC",
    }
    for metric, suffix in primary_names.items():
        baseline, candidate = metric_pair(primary, metric)
        macros[f"PrimaryDecision{suffix}"] = fmt(baseline)
        macros[f"PrimaryTraj{suffix}"] = fmt(candidate)

    repeated_names = {
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "aurc": "AURC",
    }
    for metric, suffix in repeated_names.items():
        node = repeated["macro"][metric]
        # Candidate is TrajUQ; baseline is repeated sampling.
        macros[f"Repeated{suffix}"] = fmt(node["baseline_macro"])
        macros[f"RepeatedTraj{suffix}"] = fmt(node["candidate_macro"])

    a = adaptive["macro"]
    macros.update(
        {
            "AdaptiveBaseAccuracy": fmt(a["base_accuracy"]),
            "AdaptiveBaseCost": "1.000",
            "AdaptiveRefinedAccuracy": fmt(a["refined_accuracy"]),
            "AdaptiveRefinedCost": fmt(a["refined_cost_ratio_to_base"]),
            "AdaptiveRandomAccuracy": fmt(a["random_expected_accuracy"]),
            "AdaptiveRandomCost": fmt(a["nominal_cost_ratio_to_base"]),
            "AdaptiveAccuracy": fmt(a["adaptive_accuracy"]),
            "AdaptiveCost": fmt(a["nominal_cost_ratio_to_base"]),
            "AdaptiveDeltaRandom": fmt_signed(a["delta_vs_random"]),
        }
    )

    observed = claim["observed"]
    delta = fmt_signed(observed["macro_auroc_improvement"])
    ci_low = fmt_signed(observed["macro_auroc_ci_low"])
    ci_high = fmt_signed(observed["macro_auroc_ci_high"])
    positive_families = int(observed["positive_model_families"])
    n_families = int(observed["n_model_families"])

    if claim["claim_supported"]:
        abstract_sentence = (
            "Across the frozen 18-cell matrix, trajectory features improve macro "
            f"AUROC over decision-time entropy by {delta} "
            f"(95\\% CI [{ci_low}, {ci_high}]), with positive point improvement "
            f"in {positive_families}/{n_families} model families."
        )
        results_sentence = (
            "The pre-registered RQ1 positive-claim gate is met: the 18-cell "
            f"macro AUROC improvement is {delta} (95\\% CI [{ci_low}, {ci_high}]) "
            f"and {positive_families}/{n_families} model families show positive "
            "point improvement."
        )
        conclusion_sentence = (
            f"The frozen primary analysis yields a macro AUROC improvement of {delta} "
            "over decision-time entropy and satisfies the pre-registered RQ1 claim gate."
        )
    else:
        abstract_sentence = (
            "Across the frozen 18-cell matrix, the macro AUROC difference versus "
            f"decision-time entropy is {delta} (95\\% CI [{ci_low}, {ci_high}]); "
            "the pre-registered positive-claim gate is not met."
        )
        results_sentence = (
            f"The 18-cell macro AUROC difference is {delta} "
            f"(95\\% CI [{ci_low}, {ci_high}]). The pre-registered positive-claim "
            "gate is not met, so we do not claim a consistent trajectory advantage."
        )
        conclusion_sentence = (
            "The frozen primary analysis does not satisfy the pre-registered RQ1 "
            "positive-claim gate; we therefore keep the conclusion correspondingly narrow."
        )

    macros["AbstractResultSentence"] = abstract_sentence
    macros["ResultsResultSentence"] = results_sentence
    macros["ConclusionResultSentence"] = conclusion_sentence
    return macros


def render_tex(macros: dict[str, str], sources: dict[str, tuple[str, str]]) -> str:
    lines = [
        "% GENERATED FILE — DO NOT TYPE RESULT VALUES BY HAND.",
        "% Produced by scripts/render_results_tex.py from completeness-gated artifacts.",
    ]
    for label, (path, digest) in sources.items():
        lines.append(f"% source {label}: {path} sha256={digest}")
    for name, value in macros.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render manuscript result macros from frozen aggregate artifacts."
    )
    parser.add_argument(
        "--primary",
        default="results/primary_analysis/macro_primary.json",
    )
    parser.add_argument(
        "--claim-gate",
        default="results/primary_analysis/rq1_claim_gate.json",
    )
    parser.add_argument(
        "--repeated",
        default="results/repeated_sampling/macro_trajuq_vs_repeated.json",
    )
    parser.add_argument(
        "--adaptive",
        default="results/adaptive/macro_cap_050.json",
    )
    parser.add_argument(
        "--output",
        default="paper/generated_results.tex",
    )
    parser.add_argument(
        "--manifest-output",
        default="paper/generated_results_manifest.json",
    )
    args = parser.parse_args()

    paths = {
        "primary": Path(args.primary),
        "claim_gate": Path(args.claim_gate),
        "repeated": Path(args.repeated),
        "adaptive": Path(args.adaptive),
    }
    payloads = {
        name: load_json(str(path))
        for name, path in paths.items()
    }
    macros = build_macros(
        payloads["primary"],
        payloads["claim_gate"],
        payloads["repeated"],
        payloads["adaptive"],
    )
    sources = {
        name: (str(path), sha256(path))
        for name, path in paths.items()
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_tex(macros, sources), encoding="utf-8")

    manifest = {
        "protocol": "trajuq_paper_results_render_v1",
        "output": str(output),
        "output_sha256": sha256(output),
        "sources": {
            name: {"path": path, "sha256": digest}
            for name, (path, digest) in sources.items()
        },
        "macro_count": len(macros),
        "rq1_claim_decision": payloads["claim_gate"]["decision"],
    }
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
