#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

from scripts.evaluate_rq1_claim_gate import evaluate_gate


ABLATION_SUFFIXES = {
    "decision_plus_distribution": "Distribution",
    "decision_plus_stability": "Stability",
    "decision_plus_persistence": "Persistence",
    "full": "Full",
    "full_without_js": "NoJS",
    "full_without_revisions": "NoRevisions",
    "full_without_persistence": "NoPersistence",
}


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
    if primary.get("protocol") != "hierarchical_within_cell_macro_bootstrap_v1":
        raise ValueError("Primary macro protocol is not recognized.")
    if primary.get("cell_set_complete") is not True:
        raise ValueError("Primary macro does not contain the complete frozen cell set.")
    if int(primary.get("expected_cell_count", -1)) != 18:
        raise ValueError("Primary macro expected cell count is not 18.")
    if int(primary.get("observed_cell_count", -1)) != 18:
        raise ValueError("Primary macro observed cell count is not 18.")
    for metric in ("auroc", "auprc", "brier", "ece15", "aurc"):
        metric_pair(primary, metric)


def require_claim_gate(
    claim: dict,
    primary: dict,
    primary_cfg: dict | None = None,
) -> None:
    if claim.get("protocol") != "trajuq_rq1_claim_gate_v1":
        raise ValueError("RQ1 claim gate protocol is not recognized.")

    decision = str(claim.get("decision", ""))
    if decision not in {
        "SUPPORT_POSITIVE_RQ1_CLAIM",
        "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM",
    }:
        raise ValueError("RQ1 claim gate has no valid frozen decision.")

    supported = claim.get("claim_supported")
    if not isinstance(supported, bool):
        raise ValueError("RQ1 claim gate has no boolean claim_supported value.")
    expected_decision = (
        "SUPPORT_POSITIVE_RQ1_CLAIM"
        if supported
        else "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM"
    )
    if decision != expected_decision:
        raise ValueError("RQ1 claim_supported contradicts the frozen decision.")

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
        (
            float(observed["positive_model_families"]),
            float(primary_auroc["positive_model_families"]),
        ),
        (
            float(observed["n_model_families"]),
            float(primary_auroc["n_model_families"]),
        ),
    ]
    if any(abs(left - right) > 1e-12 for left, right in checks):
        raise ValueError("RQ1 claim gate does not match the supplied primary macro.")

    if primary_cfg is not None:
        expected = evaluate_gate(primary, primary_cfg)
        for key in (
            "primary_endpoint",
            "decision",
            "claim_supported",
            "frozen_rules",
            "conditions",
        ):
            if claim.get(key) != expected.get(key):
                raise ValueError(
                    f"RQ1 claim gate field {key!r} differs from re-evaluated frozen gate."
                )
        if claim.get("observed") != expected.get("observed"):
            raise ValueError(
                "RQ1 claim gate observations differ from re-evaluated frozen gate."
            )


def require_repeated(repeated: dict) -> None:
    if repeated.get("protocol") != "trajuq_vs_repeated_macro_v2":
        raise ValueError("Repeated-sampling macro protocol is not recognized.")
    if repeated.get("comparison") != "single_trajectory_trajuq_vs_five_sample_baseline":
        raise ValueError("Repeated-sampling comparison contract is not recognized.")
    if repeated.get("cell_set_complete") is not True:
        raise ValueError("Repeated-sampling macro cell set is incomplete.")
    if int(repeated.get("expected_cells", -1)) != 18:
        raise ValueError("Repeated-sampling macro expected cell count is not 18.")
    if int(repeated.get("observed_cells", -1)) != 18:
        raise ValueError("Repeated-sampling macro observed cell count is not 18.")

    compute = repeated.get("auxiliary_compute", {})
    if compute.get("aggregation") != "equal_cell_weight_macro":
        raise ValueError("Repeated-sampling compute aggregation is not frozen.")
    if compute.get("complete") is not True:
        raise ValueError("Repeated-sampling compute accounting is incomplete.")
    if int(compute.get("expected_cells", -1)) != 18:
        raise ValueError("Repeated-sampling compute expected cell count is not 18.")
    if int(compute.get("observed_cells_with_complete_test_wall_time", -1)) != 18:
        raise ValueError("Repeated-sampling compute coverage is incomplete.")
    if int(compute.get("sample_count", -1)) != 5:
        raise ValueError("Repeated-sampling compute accounting is not five-sample.")

    for key in (
        "mean_test_auxiliary_wall_time_seconds_per_example",
        "mean_test_p90_auxiliary_wall_time_seconds_per_example",
    ):
        value = float(compute[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Repeated-sampling compute field {key} is invalid.")

    for metric in ("auroc", "auprc", "aurc"):
        node = repeated.get("macro", {}).get(metric)
        if not isinstance(node, dict):
            raise ValueError(f"Repeated macro missing metric={metric}")
        float(node["baseline_macro"])
        float(node["candidate_macro"])


def require_ablation(ablation: dict) -> None:
    if ablation.get("protocol") != "trajuq_feature_ablation_macro_v1":
        raise ValueError("Feature-ablation macro protocol is not recognized.")
    if ablation.get("baseline") != "decision_only":
        raise ValueError("Feature-ablation baseline is not decision_only.")
    if ablation.get("cell_set_complete") is not True:
        raise ValueError("Feature-ablation macro cell set is incomplete.")
    if int(ablation.get("expected_cells", -1)) != 18:
        raise ValueError("Feature-ablation expected cell count is not 18.")
    if int(ablation.get("observed_cells", -1)) != 18:
        raise ValueError("Feature-ablation observed cell count is not 18.")
    groups = ablation.get("groups", {})
    if set(groups) != set(ABLATION_SUFFIXES):
        raise ValueError("Feature-ablation group set differs from frozen contract.")
    for group in ABLATION_SUFFIXES:
        metrics = groups[group].get("metrics", {})
        for metric in ("auroc", "auprc", "aurc"):
            node = metrics.get(metric)
            if not isinstance(node, dict):
                raise ValueError(
                    f"Feature-ablation group={group} missing metric={metric}"
                )
            for key in (
                "candidate_macro",
                "baseline_macro",
                "improvement_macro",
                "improvement_ci_low",
                "improvement_ci_high",
            ):
                value = float(node[key])
                if not math.isfinite(value):
                    raise ValueError(
                        f"Feature-ablation {group}/{metric}/{key} is non-finite."
                    )


def require_adaptive(adaptive: dict) -> None:
    if adaptive.get("protocol") != "trajuq_adaptive_macro_v1":
        raise ValueError("Adaptive macro protocol is not recognized.")
    if adaptive.get("execution_semantics") != "selective_second_pass_rerun":
        raise ValueError("Adaptive execution semantics are not the frozen protocol.")
    if adaptive.get("fraction_key") != "cap_050":
        raise ValueError("Adaptive artifact is not the frozen primary cap key.")
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
    ablation: dict,
    adaptive: dict,
    primary_cfg: dict | None = None,
) -> dict[str, str]:
    require_primary(primary)
    require_claim_gate(claim, primary, primary_cfg)
    require_repeated(repeated)
    require_ablation(ablation)
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
    repeated_seconds = float(
        repeated["auxiliary_compute"][
            "mean_test_auxiliary_wall_time_seconds_per_example"
        ]
    )
    macros["RepeatedAuxWallTime"] = f"{repeated_seconds:.2f}"

    for group, suffix in ABLATION_SUFFIXES.items():
        metrics = ablation["groups"][group]["metrics"]
        auroc = metrics["auroc"]
        macros[f"Ablation{suffix}AUROCDelta"] = fmt_signed(
            auroc["improvement_macro"]
        )
        macros[f"Ablation{suffix}AUROCLow"] = fmt_signed(
            auroc["improvement_ci_low"]
        )
        macros[f"Ablation{suffix}AUROCHigh"] = fmt_signed(
            auroc["improvement_ci_high"]
        )
        macros[f"Ablation{suffix}AUPRCDelta"] = fmt_signed(
            metrics["auprc"]["improvement_macro"]
        )
        macros[f"Ablation{suffix}AURCDelta"] = fmt_signed(
            metrics["aurc"]["improvement_macro"]
        )

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
        "--primary-config",
        default="configs/experiments/primary.yaml",
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
        "--ablation",
        default="results/primary_analysis/macro_feature_ablation.json",
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
        "ablation": Path(args.ablation),
        "adaptive": Path(args.adaptive),
    }
    payloads = {
        name: load_json(str(path))
        for name, path in paths.items()
    }
    primary_config_path = Path(args.primary_config)
    if not primary_config_path.exists():
        raise SystemExit(f"Missing primary config: {primary_config_path}")
    primary_cfg = yaml.safe_load(
        primary_config_path.read_text(encoding="utf-8")
    )
    macros = build_macros(
        payloads["primary"],
        payloads["claim_gate"],
        payloads["repeated"],
        payloads["ablation"],
        payloads["adaptive"],
        primary_cfg,
    )
    sources = {
        name: (str(path), sha256(path))
        for name, path in paths.items()
    }
    sources["primary_config"] = (
        str(primary_config_path),
        sha256(primary_config_path),
    )

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
