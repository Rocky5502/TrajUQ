#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_rq1_claim_gate import evaluate_gate  # noqa: E402


PRIOR_SIGNAL_SUFFIXES = {
    "trajectory_entropy": "PriorEntropy",
    "top1_revision_rate": "PriorFlipRate",
}


ABLATION_SUFFIXES = {
    "decision_plus_distribution": "Distribution",
    "decision_plus_stability": "Stability",
    "decision_plus_persistence": "Persistence",
    "full": "Full",
    "full_without_js": "NoJS",
    "full_without_revisions": "NoRevisions",
    "full_without_persistence": "NoPersistence",
}


CALIBRATION_SENSITIVITY_METRICS = (
    "brier",
    "nll",
    "ece",
    "ece_10",
    "ece_20",
    "aurc",
    "risk_at_50_coverage",
    "risk_at_80_coverage",
)

CALIBRATION_RENDER_METRICS = {
    "brier": "Brier",
    "nll": "NLL",
    "ece": "ECE",
    "aurc": "AURC",
}

CALIBRATION_RENDER_VIEWS = {
    "decision_entropy": "Decision",
    "trajectory_plus_decision": "Traj",
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


def require_bound_result_provenance(
    payload: dict,
    *,
    label: str,
) -> str:
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{label} has no source-binding provenance.")
    if provenance.get("protocol") != "trajuq_analysis_source_binding_v1":
        raise ValueError(f"{label} source-binding protocol is not recognized.")
    if provenance.get("git_clean") is not True:
        raise ValueError(f"{label} was not produced from a clean checkout.")
    commit = str(provenance.get("git_commit", ""))
    if not commit:
        raise ValueError(f"{label} has no frozen experiment commit.")

    sources = provenance.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError(f"{label} source-binding has no upstream files.")
    for name, spec in sources.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{label} provenance entry {name!r} is invalid.")
        path = Path(str(spec.get("path", "")))
        expected = str(spec.get("sha256", ""))
        if not path.is_file() or not expected:
            raise ValueError(
                f"{label} provenance source {name!r} is missing."
            )
        if sha256(path) != expected:
            raise ValueError(
                f"{label} provenance source hash changed: {name}."
            )
    return commit


def require_render_source_provenance(
    payloads: dict[str, dict],
    paths: dict[str, Path],
    primary_config_path: Path,
) -> None:
    commits = {
        require_bound_result_provenance(
            payloads[label],
            label=label,
        )
        for label in (
            "primary",
            "claim_gate",
            "repeated",
            "ablation",
            "prior_signals",
            "calibration_sensitivity",
            "adaptive",
        )
    }
    if len(commits) != 1:
        raise ValueError(
            "Primary-analysis render sources disagree on the frozen experiment commit."
        )

    config_hash = sha256(primary_config_path)
    for label in ("primary", "claim_gate", "calibration_sensitivity"):
        spec = payloads[label]["provenance"]["sources"].get(
            "primary_config"
        )
        if not isinstance(spec, dict):
            raise ValueError(
                f"{label} provenance does not bind the primary config."
            )
        if (
            str(Path(str(spec.get("path", ""))))
            != str(primary_config_path)
            or str(spec.get("sha256", "")) != config_hash
        ):
            raise ValueError(
                f"{label} provenance binds a different primary config."
            )

    claim_macro = payloads["claim_gate"]["provenance"]["sources"].get(
        "macro_primary"
    )
    if not isinstance(claim_macro, dict):
        raise ValueError(
            "RQ1 claim gate provenance does not bind the primary macro."
        )
    primary_path = paths["primary"]
    if (
        str(Path(str(claim_macro.get("path", ""))))
        != str(primary_path)
        or str(claim_macro.get("sha256", "")) != sha256(primary_path)
    ):
        raise ValueError(
            "RQ1 claim gate provenance does not match the supplied primary macro."
        )


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


def require_repeated(
    repeated: dict,
    primary_cfg: dict | None = None,
) -> None:
    if repeated.get("protocol") != "trajuq_vs_repeated_macro_v2":
        raise ValueError("Repeated-sampling macro protocol is not recognized.")
    if repeated.get("comparison") != "single_trajectory_trajuq_vs_five_sample_baseline":
        raise ValueError("Repeated-sampling comparison contract is not recognized.")
    if repeated.get("cell_set_complete") is not True:
        raise ValueError("Repeated-sampling macro cell set is incomplete.")

    expected_cells = int(repeated.get("expected_cells", -1))
    observed_cells = int(repeated.get("observed_cells", -1))
    if expected_cells <= 0:
        raise ValueError("Repeated-sampling macro expected cell count is invalid.")
    if observed_cells != expected_cells:
        raise ValueError("Repeated-sampling macro observed cell count is incomplete.")

    eligible = repeated.get("eligible_datasets")
    if not isinstance(eligible, list) or not eligible:
        raise ValueError("Repeated-sampling macro has no frozen eligible dataset set.")
    eligible = [str(name) for name in eligible]
    if len(eligible) != len(set(eligible)):
        raise ValueError("Repeated-sampling eligible dataset set contains duplicates.")

    if primary_cfg is not None:
        repeated_cfg = (
            primary_cfg.get("baselines", {})
            .get("expensive", {})
            .get("repeated_sampling", {})
        )
        cfg_eligible = [
            str(name)
            for name in repeated_cfg.get("eligible_datasets", [])
        ]
        if eligible != cfg_eligible:
            raise ValueError(
                "Repeated-sampling eligible datasets differ from frozen primary config."
            )
        model_count = len(
            primary_cfg.get("models", {}).get("primary", [])
        )
        cfg_expected = model_count * len(cfg_eligible)
        if expected_cells != cfg_expected:
            raise ValueError(
                "Repeated-sampling expected cell count differs from frozen config."
            )

    compute = repeated.get("auxiliary_compute", {})
    if compute.get("aggregation") != "equal_cell_weight_macro":
        raise ValueError("Repeated-sampling compute aggregation is not frozen.")
    if compute.get("complete") is not True:
        raise ValueError("Repeated-sampling compute accounting is incomplete.")
    if int(compute.get("expected_cells", -1)) != expected_cells:
        raise ValueError("Repeated-sampling compute expected cell count drifted.")
    if int(
        compute.get("observed_cells_with_complete_test_wall_time", -1)
    ) != expected_cells:
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


def require_prior_signals(prior: dict) -> None:
    if prior.get("protocol") != "trajuq_prior_aligned_signal_macro_v1":
        raise ValueError("Prior-signal macro protocol is not recognized.")
    if (
        prior.get("role")
        != "secondary_prior_aligned_signal_class_diagnostic"
    ):
        raise ValueError("Prior-signal macro role is not recognized.")
    if prior.get("cell_set_complete") is not True:
        raise ValueError("Prior-signal macro cell set is incomplete.")
    if int(prior.get("expected_cells", -1)) != 18:
        raise ValueError("Prior-signal expected cell count is not 18.")
    if int(prior.get("observed_cells", -1)) != 18:
        raise ValueError("Prior-signal observed cell count is not 18.")
    macro = prior.get("macro", {})
    if set(macro) != set(PRIOR_SIGNAL_SUFFIXES):
        raise ValueError("Prior-signal set differs from frozen contract.")
    for signal in PRIOR_SIGNAL_SUFFIXES:
        metrics = macro[signal]
        for metric in ("auroc", "auprc", "aurc"):
            node = metrics.get(metric)
            if not isinstance(node, dict):
                raise ValueError(
                    f"Prior-signal {signal} missing metric={metric}"
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
                        f"Prior-signal {signal}/{metric}/{key} is non-finite."
                    )


def require_calibration_sensitivity(sensitivity: dict) -> None:
    if (
        sensitivity.get("protocol")
        != "trajuq_isotonic_calibration_sensitivity_macro_v1"
    ):
        raise ValueError(
            "Calibration-sensitivity macro protocol is not recognized."
        )
    if (
        sensitivity.get("role")
        != "supplementary_calibration_sensitivity"
    ):
        raise ValueError("Calibration-sensitivity role is not recognized.")
    if sensitivity.get("cell_set_complete") is not True:
        raise ValueError("Calibration-sensitivity cell set is incomplete.")
    if int(sensitivity.get("expected_cells", -1)) != 18:
        raise ValueError(
            "Calibration-sensitivity expected cell count is not 18."
        )
    if int(sensitivity.get("observed_cells", -1)) != 18:
        raise ValueError(
            "Calibration-sensitivity observed cell count is not 18."
        )

    eligible = set(sensitivity.get("eligible_cells", []))
    skipped = set(sensitivity.get("skipped_cells", []))
    per_cell = sensitivity.get("per_cell", {})
    if eligible.intersection(skipped):
        raise ValueError(
            "Calibration-sensitivity eligible/skipped cell sets overlap."
        )
    if len(eligible) != int(sensitivity.get("eligible_cell_count", -1)):
        raise ValueError(
            "Calibration-sensitivity eligible-cell count is inconsistent."
        )
    if len(skipped) != int(sensitivity.get("skipped_cell_count", -1)):
        raise ValueError(
            "Calibration-sensitivity skipped-cell count is inconsistent."
        )
    labels = eligible | skipped
    if len(labels) != 18 or set(per_cell) != labels:
        raise ValueError(
            "Calibration-sensitivity does not partition the full 18-cell set."
        )

    if (
        sensitivity.get("aggregation")
        != (
            "equal_weight_over_pre_registered_eligible_cells_only; all frozen "
            "ineligible cells remain explicit skips"
        )
    ):
        raise ValueError(
            "Calibration-sensitivity aggregation policy differs from frozen contract."
        )
    if (
        sensitivity.get("delta_direction")
        != (
            "positive means isotonic improves a lower-is-better metric "
            "relative to the primary logistic calibrator"
        )
    ):
        raise ValueError(
            "Calibration-sensitivity delta direction differs from frozen contract."
        )
    if sensitivity.get("metrics") != list(CALIBRATION_SENSITIVITY_METRICS):
        raise ValueError(
            "Calibration-sensitivity declared metric list differs from frozen contract."
        )

    provenance_sources = sensitivity.get("provenance", {}).get("sources", {})
    expected_cell_sources = {f"cell:{label}" for label in labels}
    observed_cell_sources = {
        str(name)
        for name in provenance_sources
        if str(name).startswith("cell:")
    }
    if observed_cell_sources != expected_cell_sources:
        raise ValueError(
            "Calibration-sensitivity provenance cell set differs from rendered cells."
        )

    policy = sensitivity.get("eligibility_policy", {})
    expected_policy = {
        "role": "supplementary_calibration_sensitivity",
        "base_estimator": "standardized_logistic",
        "cv_folds": 5,
        "ensemble": False,
        "min_calibration_samples": 250,
        "min_class_count": 25,
    }
    if policy != expected_policy:
        raise ValueError(
            "Calibration-sensitivity eligibility policy differs from frozen contract."
        )

    macro = sensitivity.get("macro_over_eligible_cells", {})
    if set(macro) != set(CALIBRATION_RENDER_VIEWS):
        raise ValueError(
            "Calibration-sensitivity view set differs from frozen contract."
        )
    for view in CALIBRATION_RENDER_VIEWS:
        metrics = macro[view]
        if set(metrics) != set(CALIBRATION_SENSITIVITY_METRICS):
            raise ValueError(
                f"Calibration-sensitivity metric set differs for view={view}."
            )
        for metric in CALIBRATION_SENSITIVITY_METRICS:
            node = metrics[metric]
            if set(node) != {
                "primary_logistic",
                "isotonic_sensitivity",
                "isotonic_minus_logistic_improvement",
            }:
                raise ValueError(
                    f"Calibration-sensitivity fields differ for {view}/{metric}."
                )
            for value in node.values():
                if eligible:
                    number = float(value)
                    if not math.isfinite(number):
                        raise ValueError(
                            f"Calibration-sensitivity {view}/{metric} is non-finite."
                        )
                elif value is not None:
                    raise ValueError(
                        "Calibration-sensitivity macro must use null metrics "
                        "when no cell is eligible."
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


def fmt_optional(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "--"
    return fmt_signed(value) if signed else fmt(value)


def build_macros(
    primary: dict,
    claim: dict,
    repeated: dict,
    ablation: dict,
    prior_signals: dict,
    calibration_sensitivity: dict,
    adaptive: dict,
    primary_cfg: dict | None = None,
) -> dict[str, str]:
    require_primary(primary)
    require_claim_gate(claim, primary, primary_cfg)
    require_repeated(repeated, primary_cfg)
    require_ablation(ablation)
    require_prior_signals(prior_signals)
    require_calibration_sensitivity(calibration_sensitivity)
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
    macros["RepeatedCellCount"] = str(int(repeated["expected_cells"]))
    macros["RepeatedDatasetCount"] = str(
        len(repeated["eligible_datasets"])
    )

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

    for signal, suffix in PRIOR_SIGNAL_SUFFIXES.items():
        metrics = prior_signals["macro"][signal]
        auroc = metrics["auroc"]
        macros[f"{suffix}AUROC"] = fmt(auroc["baseline_macro"])
        macros[f"{suffix}TrajAUROC"] = fmt(auroc["candidate_macro"])
        macros[f"{suffix}AUROCDelta"] = fmt_signed(
            auroc["improvement_macro"]
        )
        macros[f"{suffix}AUROCLow"] = fmt_signed(
            auroc["improvement_ci_low"]
        )
        macros[f"{suffix}AUROCHigh"] = fmt_signed(
            auroc["improvement_ci_high"]
        )
        macros[f"{suffix}AUPRCDelta"] = fmt_signed(
            metrics["auprc"]["improvement_macro"]
        )
        macros[f"{suffix}AURCDelta"] = fmt_signed(
            metrics["aurc"]["improvement_macro"]
        )

    macros["CalibrationEligibleCells"] = str(
        int(calibration_sensitivity["eligible_cell_count"])
    )
    macros["CalibrationSkippedCells"] = str(
        int(calibration_sensitivity["skipped_cell_count"])
    )
    sensitivity_macro = calibration_sensitivity[
        "macro_over_eligible_cells"
    ]
    for view, view_suffix in CALIBRATION_RENDER_VIEWS.items():
        for metric, metric_suffix in CALIBRATION_RENDER_METRICS.items():
            node = sensitivity_macro[view][metric]
            macros[
                f"Calibration{view_suffix}{metric_suffix}Logistic"
            ] = fmt_optional(node["primary_logistic"])
            macros[
                f"Calibration{view_suffix}{metric_suffix}Isotonic"
            ] = fmt_optional(node["isotonic_sensitivity"])
            macros[
                f"Calibration{view_suffix}{metric_suffix}Delta"
            ] = fmt_optional(
                node["isotonic_minus_logistic_improvement"],
                signed=True,
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
    macros["PaperStateLimitationsSentence"] = ""
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
        "--prior-signals",
        default="results/primary_analysis/macro_prior_signal_diagnostics.json",
    )
    parser.add_argument(
        "--calibration-sensitivity",
        default="results/primary_analysis/macro_calibration_sensitivity.json",
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
        "prior_signals": Path(args.prior_signals),
        "calibration_sensitivity": Path(args.calibration_sensitivity),
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
    require_render_source_provenance(
        payloads,
        paths,
        primary_config_path,
    )
    macros = build_macros(
        payloads["primary"],
        payloads["claim_gate"],
        payloads["repeated"],
        payloads["ablation"],
        payloads["prior_signals"],
        payloads["calibration_sensitivity"],
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
        "protocol": "trajuq_paper_results_render_v2",
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
