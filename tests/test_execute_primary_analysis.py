import hashlib
import json
from pathlib import Path

import scripts.execute_primary_analysis as execute_primary_analysis
from trajuq.features import FROZEN_FEATURE_GROUPS
from scripts.execute_primary_analysis import (
    aggregate_provenance_matches,
    claim_matches_frozen_rule,
    execution_snapshot,
    output_valid,
    semantic_output_valid,
    step_state,
    validate_plan_provenance,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _analysis_binding(
    commit: str,
    calibration: Path,
    test: Path,
) -> dict:
    return {
        "protocol": "trajuq_analysis_source_binding_v1",
        "git_commit": commit,
        "git_clean": True,
        "sources": {
            "calibration": {
                "path": str(calibration),
                "sha256": _sha256(calibration),
            },
            "test": {
                "path": str(test),
                "sha256": _sha256(test),
            },
        },
    }


def _test_only_binding(commit: str, test: Path) -> dict:
    return {
        "protocol": "trajuq_analysis_source_binding_v1",
        "git_commit": commit,
        "git_clean": True,
        "sources": {
            "test": {
                "path": str(test),
                "sha256": _sha256(test),
            },
        },
    }


def _row_binding(provenance: dict) -> dict:
    return {
        "protocol": provenance["protocol"],
        "git_commit": provenance["git_commit"],
        "git_clean": provenance["git_clean"],
        "source_sha256": {
            name: spec["sha256"]
            for name, spec in provenance["sources"].items()
        },
    }


def test_output_valid_json_and_jsonl(tmp_path):
    metrics = tmp_path / "metrics.json"
    predictions = tmp_path / "predictions.jsonl"
    metrics.write_text('{"auroc": 0.5}', encoding="utf-8")
    predictions.write_text('{"example_id":"x"}\n', encoding="utf-8")
    assert output_valid(metrics)
    assert output_valid(predictions)


def test_evaluate_step_detects_partial_outputs(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"auroc": 0.5}', encoding="utf-8")
    cell = {
        "outputs": {
            "metrics": str(metrics),
            "predictions": str(tmp_path / "predictions.jsonl"),
        }
    }
    assert step_state(cell, ("metrics", "predictions")) == "PARTIAL"


def test_semantic_output_rejects_truncated_predictions(tmp_path):
    cal = tmp_path / "calibration.scored.jsonl"
    test = tmp_path / "test.scored.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    cal.write_text(
        '{"example_id":"c1"}\n{"example_id":"c2"}\n',
        encoding="utf-8",
    )
    test.write_text(
        '{"example_id":"t1"}\n{"example_id":"t2"}\n',
        encoding="utf-8",
    )
    predictions.write_text(
        '{"example_id":"t1"}\n',
        encoding="utf-8",
    )
    cell = {
        "inputs": {
            "calibration": {"path": str(cal)},
            "test": {"path": str(test)},
        }
    }

    assert not semantic_output_valid(
        cell,
        "predictions",
        predictions,
    )


def test_analysis_snapshot_requires_macro_and_claim_gate(tmp_path):
    commit = "a" * 40
    cal = tmp_path / "calibration.scored.jsonl"
    test = tmp_path / "test.scored.jsonl"
    cal.write_text(
        '{"example_id":"c","correct":true}\n',
        encoding="utf-8",
    )
    test.write_text(
        '{"example_id":"x","correct":false}\n',
        encoding="utf-8",
    )
    provenance = _analysis_binding(commit, cal, test)
    row_binding = _row_binding(provenance)
    test_only_provenance = _test_only_binding(commit, test)

    outputs = {
        "metrics": tmp_path / "metrics.json",
        "predictions": tmp_path / "predictions.jsonl",
        "nested_bootstrap": tmp_path / "nested.json",
        "feature_ablation": tmp_path / "ablation.json",
        "feature_ablation_predictions": tmp_path / "ablation_predictions.jsonl",
        "prior_signal_diagnostic": tmp_path / "prior_signal.json",
        "prior_signal_predictions": tmp_path / "prior_signal_predictions.jsonl",
        "calibration_sensitivity": tmp_path / "calibration_sensitivity.json",
        "raw_risk_diagnostic": tmp_path / "raw.json",
    }
    outputs["metrics"].write_text(
        json.dumps(
            {
                "protocol": "trajuq_heldout_evaluation_v1",
                "provenance": provenance,
                "decision_entropy_baseline": {
                    "n_calibration": 1,
                    "n_test": 1,
                },
                "trajectory_plus_decision_logistic": {
                    "n_calibration": 1,
                    "n_test": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    outputs["predictions"].write_text(
        json.dumps(
            {
                "example_id": "x",
                "analysis_provenance": row_binding,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["nested_bootstrap"].write_text(
        json.dumps(
            {
                "protocol": "trajuq_nested_calibrated_comparison_v1",
                "provenance": provenance,
                "n_calibration": 1,
                "n_test": 1,
                "metrics": {"auroc": {}, "auprc": {}, "aurc": {}},
            }
        ),
        encoding="utf-8",
    )
    outputs["feature_ablation"].write_text(
        json.dumps(
            {
                "protocol": "frozen_feature_family_ablation_v1",
                "provenance": provenance,
                "baseline": "decision_only",
                "group_order": list(FROZEN_FEATURE_GROUPS),
                "n_calibration": 1,
                "n_test": 1,
                "groups": {
                    name: {}
                    for name in FROZEN_FEATURE_GROUPS
                },
                "paired_improvements_vs_decision_only": {
                    name: {}
                    for name in FROZEN_FEATURE_GROUPS
                    if name != "decision_only"
                },
            }
        ),
        encoding="utf-8",
    )
    outputs["feature_ablation_predictions"].write_text(
        json.dumps(
            {
                "example_id": "x",
                "y_error": 1,
                "correct": False,
                "p_error_by_group": {
                    name: 0.5
                    for name in FROZEN_FEATURE_GROUPS
                },
                "analysis_provenance": row_binding,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["prior_signal_diagnostic"].write_text(
        json.dumps(
            {
                "protocol": "trajuq_prior_aligned_signal_diagnostic_v1",
                "provenance": provenance,
                "role": "secondary_prior_aligned_signal_class_diagnostic",
                "signal_order": [
                    "trajectory_entropy",
                    "top1_revision_rate",
                ],
                "n_calibration": 1,
                "n_test": 1,
                "signals": {
                    "trajectory_entropy": {
                        "n_calibration": 1,
                        "n_test": 1,
                    },
                    "top1_revision_rate": {
                        "n_calibration": 1,
                        "n_test": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    outputs["prior_signal_predictions"].write_text(
        json.dumps(
            {
                "example_id": "x",
                "y_error": 1,
                "correct": False,
                "p_error_trajectory_entropy": 0.6,
                "p_error_top1_revision_rate": 0.7,
                "analysis_provenance": row_binding,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    outputs["calibration_sensitivity"].write_text(
        json.dumps(
            {
                "protocol": "trajuq_isotonic_calibration_sensitivity_v1",
                "role": "supplementary_calibration_sensitivity",
                "status": "SKIP_INSUFFICIENT_CALIBRATION",
                "provenance": provenance,
                "selective_risk_version": "tie_aware_uniform_tie_break_v2",
                "seed": 2027,
                "n_calibration": 1,
                "n_test": 1,
                "eligibility": {
                    "eligible": False,
                    "min_calibration_samples_required": 1000,
                    "min_class_count_required": 5,
                    "cv_folds_requested": 5,
                    "observed_class_counts": {"0": 1},
                    "observed_min_class_count": 1,
                },
                "method": {
                    "base_estimator": "standardized_logistic",
                    "isotonic_cv": "stratified_out_of_fold",
                    "ensemble": False,
                    "test_labels_used_for_fit": False,
                },
            }
        ),
        encoding="utf-8",
    )
    outputs["raw_risk_diagnostic"].write_text(
        json.dumps(
            {
                "protocol": "trajuq_raw_risk_diagnostic_v1",
                "provenance": test_only_provenance,
                "n": 1,
                "metrics": {"auroc": {}, "auprc": {}, "aurc": {}},
            }
        ),
        encoding="utf-8",
    )

    macro = tmp_path / "macro.json"
    ablation_macro = tmp_path / "ablation_macro.json"
    prior_macro = tmp_path / "prior_macro.json"
    claim = tmp_path / "claim.json"
    plan = {
        "status": "PASS",
        "cell_count": 1,
        "cell_analyses": [
            {
                "cell": "llada:gsm8k",
                "adapter": "llada",
                "dataset": "gsm8k",
                "experiment_git_commit": commit,
                "calibration_sensitivity_contract": {
                    "role": "supplementary_calibration_sensitivity",
                    "base_estimator": "standardized_logistic",
                    "cv_folds": 5,
                    "ensemble": False,
                    "min_calibration_samples": 1000,
                    "min_class_count": 5,
                },
                "inputs": {
                    "calibration": {
                        "path": str(cal),
                        "sha256": _sha256(cal),
                    },
                    "test": {
                        "path": str(test),
                        "sha256": _sha256(test),
                    },
                },
                "outputs": {key: str(value) for key, value in outputs.items()},
            }
        ],
        "final_outputs": {
            "macro_primary": str(macro),
            "macro_feature_ablation": str(ablation_macro),
            "macro_prior_signal_diagnostics": str(prior_macro),
            "rq1_claim_gate": str(claim),
        },
    }

    snapshot = execution_snapshot(plan)
    assert snapshot["complete_cells"] == 1
    assert snapshot["status"] == "IN_PROGRESS"

    macro.write_text(
        json.dumps(
            {
                "protocol": "hierarchical_within_cell_macro_bootstrap_v1",
                "cell_set_complete": True,
                "expected_cell_count": 1,
                "observed_cell_count": 1,
                "macro": {
                    key: {}
                    for key in (
                        "auroc",
                        "auprc",
                        "brier",
                        "nll",
                        "ece15",
                        "aurc",
                        "risk50",
                        "risk80",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    ablation_macro.write_text(
        json.dumps(
            {
                "protocol": "trajuq_feature_ablation_macro_v1",
                "baseline": "decision_only",
                "cell_set_complete": True,
                "expected_cells": 1,
                "observed_cells": 1,
                "groups": {
                    name: {
                        "metrics": {
                            "auroc": {},
                            "auprc": {},
                            "aurc": {},
                        }
                    }
                    for name in FROZEN_FEATURE_GROUPS
                    if name != "decision_only"
                },
            }
        ),
        encoding="utf-8",
    )
    prior_macro.write_text(
        json.dumps(
            {
                "protocol": "trajuq_prior_aligned_signal_macro_v1",
                "role": "secondary_prior_aligned_signal_class_diagnostic",
                "cell_set_complete": True,
                "expected_cells": 1,
                "observed_cells": 1,
                "macro": {
                    signal: {
                        "auroc": {},
                        "auprc": {},
                        "aurc": {},
                    }
                    for signal in (
                        "trajectory_entropy",
                        "top1_revision_rate",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    claim.write_text(
        json.dumps(
            {
                "protocol": "trajuq_rq1_claim_gate_v1",
                "decision": "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM",
                "claim_supported": False,
                "conditions": {"complete_frozen_cell_set": True},
            }
        ),
        encoding="utf-8",
    )
    snapshot = execution_snapshot(plan)
    assert snapshot["status"] == "COMPLETE"


def test_claim_reuse_rejects_hand_edited_positive_decision(tmp_path):
    macro = tmp_path / "macro.json"
    config = tmp_path / "primary.yaml"
    macro.write_text(
        json.dumps(
            {
                "cell_set_complete": True,
                "macro": {
                    "auroc": {
                        "improvement_macro": 0.01,
                        "improvement_ci_low": -0.02,
                        "improvement_ci_high": 0.04,
                        "positive_model_families": 3,
                        "n_model_families": 3,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config.write_text(
        "statistics:\n"
        "  primary_endpoint: hierarchical_macro_auroc_improvement\n"
        "  rq1_claim_gate:\n"
        "    macro_auroc_ci_low_gt_zero: true\n"
        "    min_positive_model_families: 2\n",
        encoding="utf-8",
    )
    plan = {
        "protocol": "trajuq_primary_analysis_plan_v2",
        "primary_config": str(config),
        "final_outputs": {"macro_primary": str(macro)},
    }
    edited = {
        "primary_endpoint": "hierarchical_macro_auroc_improvement",
        "decision": "SUPPORT_POSITIVE_RQ1_CLAIM",
        "claim_supported": True,
        "frozen_rules": {
            "macro_auroc_ci_low_gt_zero": True,
            "min_positive_model_families": 2,
        },
        "observed": {
            "macro_auroc_improvement": 0.01,
            "macro_auroc_ci_low": -0.02,
            "macro_auroc_ci_high": 0.04,
            "positive_model_families": 3,
            "n_model_families": 3,
        },
        "conditions": {
            "complete_frozen_cell_set": True,
            "macro_auroc_improvement_positive": True,
            "macro_auroc_ci_low_gt_zero": True,
            "positive_model_families_at_least_minimum": True,
            "three_primary_model_families_present": True,
        },
    }

    assert not claim_matches_frozen_rule(plan, edited)


def test_analysis_provenance_rejects_scored_input_drift(
    tmp_path,
    monkeypatch,
):
    commit = "a" * 40
    primary_plan = tmp_path / "primary_plan.json"
    primary_config = tmp_path / "primary.yaml"
    source_input = tmp_path / "calibration.jsonl"
    raw = tmp_path / "calibration.raw.jsonl"
    raw_audit = tmp_path / "calibration.raw.jsonl.audit.json"
    scored = tmp_path / "calibration.scored.jsonl"
    audit = tmp_path / "calibration.scored.jsonl.audit.json"

    primary_plan.write_text('{"status":"PASS"}\n', encoding="utf-8")
    primary_config.write_text("experiment:\n  seed: 2027\n", encoding="utf-8")
    source_input.write_text('{"id":"x"}\n', encoding="utf-8")
    raw.write_text('{"example_id":"x"}\n', encoding="utf-8")
    raw_audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": commit,
                "git_clean": True,
                "stage": "raw",
                "sha256": _sha256(raw),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scored.write_text('{"example_id":"x"}\n', encoding="utf-8")
    audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": commit,
                "git_clean": True,
                "stage": "scored",
                "sha256": _sha256(scored),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    input_spec = {
        "source_input": str(source_input),
        "source_input_sha256": _sha256(source_input),
        "raw_path": str(raw),
        "raw_sha256": _sha256(raw),
        "raw_audit_path": str(raw_audit),
        "raw_audit_sha256": _sha256(raw_audit),
        "path": str(scored),
        "sha256": _sha256(scored),
        "audit_path": str(audit),
        "audit_sha256": _sha256(audit),
    }
    plan = {
        "protocol": "trajuq_primary_analysis_plan_v2",
        "status": "PASS",
        "experiment_git_commit": commit,
        "primary_plan": str(primary_plan),
        "primary_plan_sha256": _sha256(primary_plan),
        "primary_config": str(primary_config),
        "primary_config_sha256": _sha256(primary_config),
        "cell_analyses": [
            {
                "cell": "llada:gsm8k",
                "inputs": {
                    "calibration": dict(input_spec),
                    "test": dict(input_spec),
                },
            }
        ],
    }
    monkeypatch.setattr(execute_primary_analysis, "git_head", lambda: commit)
    monkeypatch.setattr(
        execute_primary_analysis,
        "git_status_porcelain",
        lambda: "",
    )
    monkeypatch.setattr(
        execute_primary_analysis,
        "validate_primary_plan_provenance",
        lambda payload, *, require_current_commit=True: [],
    )

    assert validate_plan_provenance(plan) == []

    scored.write_text('{"example_id":"tampered"}\n', encoding="utf-8")
    failures = validate_plan_provenance(plan)
    assert any("scored input hash differs" in item for item in failures)
    assert any("audit no longer binds scored input" in item for item in failures)


def test_analysis_provenance_rejects_audit_from_other_commit(
    tmp_path,
    monkeypatch,
):
    commit = "a" * 40
    primary_plan = tmp_path / "primary_plan.json"
    primary_config = tmp_path / "primary.yaml"
    source_input = tmp_path / "test.jsonl"
    raw = tmp_path / "test.raw.jsonl"
    raw_audit = tmp_path / "test.raw.jsonl.audit.json"
    scored = tmp_path / "test.scored.jsonl"
    audit = tmp_path / "test.scored.jsonl.audit.json"

    primary_plan.write_text('{"status":"PASS"}\n', encoding="utf-8")
    primary_config.write_text("experiment:\n  seed: 2027\n", encoding="utf-8")
    source_input.write_text('{"id":"x"}\n', encoding="utf-8")
    raw.write_text('{"example_id":"x"}\n', encoding="utf-8")
    raw_audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": commit,
                "git_clean": True,
                "stage": "raw",
                "sha256": _sha256(raw),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scored.write_text('{"example_id":"x"}\n', encoding="utf-8")
    audit.write_text(
        json.dumps(
            {
                "protocol": "trajuq_run_audit_v1",
                "status": "PASS",
                "git_commit": "other-commit",
                "git_clean": True,
                "stage": "scored",
                "sha256": _sha256(scored),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = {
        "source_input": str(source_input),
        "source_input_sha256": _sha256(source_input),
        "raw_path": str(raw),
        "raw_sha256": _sha256(raw),
        "raw_audit_path": str(raw_audit),
        "raw_audit_sha256": _sha256(raw_audit),
        "path": str(scored),
        "sha256": _sha256(scored),
        "audit_path": str(audit),
        "audit_sha256": _sha256(audit),
    }
    plan = {
        "protocol": "trajuq_primary_analysis_plan_v2",
        "status": "PASS",
        "experiment_git_commit": commit,
        "primary_plan": str(primary_plan),
        "primary_plan_sha256": _sha256(primary_plan),
        "primary_config": str(primary_config),
        "primary_config_sha256": _sha256(primary_config),
        "cell_analyses": [
            {
                "cell": "llada:gsm8k",
                "inputs": {"calibration": dict(spec), "test": dict(spec)},
            }
        ],
    }
    monkeypatch.setattr(execute_primary_analysis, "git_head", lambda: commit)
    monkeypatch.setattr(execute_primary_analysis, "git_status_porcelain", lambda: "")

    failures = validate_plan_provenance(plan)
    assert any("clean frozen scored PASS" in item for item in failures)


def test_feature_ablation_prediction_semantics_reject_group_drift(tmp_path):
    cal = tmp_path / "cal.jsonl"
    test = tmp_path / "test.jsonl"
    pred = tmp_path / "ablation_predictions.jsonl"
    cal.write_text(
        '{"example_id":"c","correct":true}\n',
        encoding="utf-8",
    )
    test.write_text(
        '{"example_id":"t","correct":false}\n',
        encoding="utf-8",
    )
    pred.write_text(
        json.dumps(
            {
                "example_id": "t",
                "y_error": 1,
                "correct": False,
                "p_error_by_group": {"decision_only": 0.5},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cell = {
        "inputs": {
            "calibration": {"path": str(cal)},
            "test": {"path": str(test)},
        }
    }
    assert not semantic_output_valid(
        cell,
        "feature_ablation_predictions",
        pred,
    )


def test_prior_signal_prediction_semantics_reject_bad_probability(tmp_path):
    cal = tmp_path / "cal.jsonl"
    test = tmp_path / "test.jsonl"
    pred = tmp_path / "prior_signal_predictions.jsonl"
    cal.write_text(
        '{"example_id":"c","correct":true}\n',
        encoding="utf-8",
    )
    test.write_text(
        '{"example_id":"t","correct":false}\n',
        encoding="utf-8",
    )
    pred.write_text(
        json.dumps(
            {
                "example_id": "t",
                "y_error": 1,
                "correct": False,
                "p_error_trajectory_entropy": 1.2,
                "p_error_top1_revision_rate": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cell = {
        "inputs": {
            "calibration": {"path": str(cal)},
            "test": {"path": str(test)},
        }
    }
    assert not semantic_output_valid(
        cell,
        "prior_signal_predictions",
        pred,
    )


def test_semantic_output_rejects_stale_commit_binding(tmp_path):
    commit = "a" * 40
    cal = tmp_path / "calibration.scored.jsonl"
    test = tmp_path / "test.scored.jsonl"
    metrics = tmp_path / "metrics.json"
    cal.write_text(
        '{"example_id":"c","correct":true}\n',
        encoding="utf-8",
    )
    test.write_text(
        '{"example_id":"t","correct":false}\n',
        encoding="utf-8",
    )
    stale = _analysis_binding("b" * 40, cal, test)
    metrics.write_text(
        json.dumps(
            {
                "protocol": "trajuq_heldout_evaluation_v1",
                "provenance": stale,
                "decision_entropy_baseline": {
                    "n_calibration": 1,
                    "n_test": 1,
                },
                "trajectory_plus_decision_logistic": {
                    "n_calibration": 1,
                    "n_test": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    cell = {
        "experiment_git_commit": commit,
        "inputs": {
            "calibration": {
                "path": str(cal),
                "sha256": _sha256(cal),
            },
            "test": {
                "path": str(test),
                "sha256": _sha256(test),
            },
        },
    }
    assert not semantic_output_valid(cell, "metrics", metrics)


def test_aggregate_provenance_rejects_prediction_drift(tmp_path):
    commit = "a" * 40
    config = tmp_path / "primary.yaml"
    prediction = tmp_path / "predictions.jsonl"
    config.write_text("experiment:\n  seed: 2027\n", encoding="utf-8")
    prediction.write_text(
        '{"example_id":"t","p_error_trajuq":0.4}\n',
        encoding="utf-8",
    )
    plan = {
        "protocol": "trajuq_primary_analysis_plan_v2",
        "experiment_git_commit": commit,
        "primary_config": str(config),
        "primary_config_sha256": _sha256(config),
        "cell_analyses": [
            {
                "cell": "llada:gsm8k",
                "outputs": {"predictions": str(prediction)},
            }
        ],
    }
    provenance = {
        "protocol": "trajuq_analysis_source_binding_v1",
        "git_commit": commit,
        "git_clean": True,
        "sources": {
            "primary_config": {
                "path": str(config),
                "sha256": _sha256(config),
            },
            "prediction:llada_gsm8k": {
                "path": str(prediction),
                "sha256": _sha256(prediction),
            },
        },
    }
    payload = {"provenance": provenance}

    assert aggregate_provenance_matches(plan, payload, "primary")

    prediction.write_text(
        '{"example_id":"t","p_error_trajuq":0.9}\n',
        encoding="utf-8",
    )
    assert not aggregate_provenance_matches(plan, payload, "primary")


def test_analysis_provenance_surfaces_nested_primary_plan_failure(
    tmp_path,
    monkeypatch,
):
    commit = "d" * 40
    primary_plan = tmp_path / "primary_plan.json"
    primary_config = tmp_path / "primary.yaml"
    primary_plan.write_text('{"status":"PASS"}\n', encoding="utf-8")
    primary_config.write_text("experiment:\n  seed: 2027\n", encoding="utf-8")

    plan = {
        "protocol": "trajuq_primary_analysis_plan_v2",
        "status": "PASS",
        "experiment_git_commit": commit,
        "primary_plan": str(primary_plan),
        "primary_plan_sha256": _sha256(primary_plan),
        "primary_config": str(primary_config),
        "primary_config_sha256": _sha256(primary_config),
        "cell_analyses": [],
    }
    monkeypatch.setattr(execute_primary_analysis, "git_head", lambda: commit)
    monkeypatch.setattr(
        execute_primary_analysis,
        "git_status_porcelain",
        lambda: "",
    )
    monkeypatch.setattr(
        execute_primary_analysis,
        "validate_primary_plan_provenance",
        lambda payload, *, require_current_commit=True: [
            "primary readiness bound source hash drifted: results/llada_parity.json"
        ],
    )

    failures = validate_plan_provenance(plan)
    assert any(
        "primary plan provenance: primary readiness bound source hash drifted"
        in item
        for item in failures
    )


def _isotonic_contract() -> dict:
    return {
        "role": "supplementary_calibration_sensitivity",
        "base_estimator": "standardized_logistic",
        "cv_folds": 5,
        "ensemble": False,
        "min_calibration_samples": 1000,
        "min_class_count": 5,
    }


def _write_scored_rows(path: Path, rows: list[tuple[str, bool]]) -> None:
    path.write_text(
        "".join(
            json.dumps({"example_id": item_id, "correct": correct}) + "\n"
            for item_id, correct in rows
        ),
        encoding="utf-8",
    )


def test_calibration_sensitivity_accepts_recomputed_insufficient_skip(tmp_path):
    commit = "a" * 40
    cal = tmp_path / "cal.scored.jsonl"
    test = tmp_path / "test.scored.jsonl"
    out = tmp_path / "sensitivity.json"
    _write_scored_rows(cal, [("c0", True), ("c1", False)])
    _write_scored_rows(test, [("t0", True)])
    provenance = _analysis_binding(commit, cal, test)

    payload = {
        "protocol": "trajuq_isotonic_calibration_sensitivity_v1",
        "role": "supplementary_calibration_sensitivity",
        "status": "SKIP_INSUFFICIENT_CALIBRATION",
        "provenance": provenance,
        "selective_risk_version": "tie_aware_uniform_tie_break_v2",
        "seed": 2027,
        "n_calibration": 2,
        "n_test": 1,
        "eligibility": {
            "eligible": False,
            "min_calibration_samples_required": 1000,
            "min_class_count_required": 5,
            "cv_folds_requested": 5,
            "observed_class_counts": {"0": 1, "1": 1},
            "observed_min_class_count": 1,
        },
        "method": {
            "base_estimator": "standardized_logistic",
            "isotonic_cv": "stratified_out_of_fold",
            "ensemble": False,
            "test_labels_used_for_fit": False,
        },
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    cell = {
        "experiment_git_commit": commit,
        "calibration_sensitivity_contract": _isotonic_contract(),
        "inputs": {
            "calibration": {"path": str(cal), "sha256": _sha256(cal)},
            "test": {"path": str(test), "sha256": _sha256(test)},
        },
    }

    assert semantic_output_valid(cell, "calibration_sensitivity", out)


def test_calibration_sensitivity_rejects_forged_skip_when_eligible(tmp_path):
    commit = "a" * 40
    cal = tmp_path / "cal.scored.jsonl"
    test = tmp_path / "test.scored.jsonl"
    out = tmp_path / "sensitivity.json"
    rows = [
        (f"c{i}", i % 2 == 0)
        for i in range(1000)
    ]
    _write_scored_rows(cal, rows)
    _write_scored_rows(test, [("t0", True), ("t1", False)])
    provenance = _analysis_binding(commit, cal, test)

    payload = {
        "protocol": "trajuq_isotonic_calibration_sensitivity_v1",
        "role": "supplementary_calibration_sensitivity",
        "status": "SKIP_INSUFFICIENT_CALIBRATION",
        "provenance": provenance,
        "selective_risk_version": "tie_aware_uniform_tie_break_v2",
        "seed": 2027,
        "n_calibration": 1000,
        "n_test": 2,
        "eligibility": {
            "eligible": False,
            "min_calibration_samples_required": 1000,
            "min_class_count_required": 5,
            "cv_folds_requested": 5,
            "observed_class_counts": {"0": 500, "1": 500},
            "observed_min_class_count": 500,
        },
        "method": {
            "base_estimator": "standardized_logistic",
            "isotonic_cv": "stratified_out_of_fold",
            "ensemble": False,
            "test_labels_used_for_fit": False,
        },
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    cell = {
        "experiment_git_commit": commit,
        "calibration_sensitivity_contract": _isotonic_contract(),
        "inputs": {
            "calibration": {"path": str(cal), "sha256": _sha256(cal)},
            "test": {"path": str(test), "sha256": _sha256(test)},
        },
    }

    assert not semantic_output_valid(cell, "calibration_sensitivity", out)


def test_calibration_sensitivity_accepts_valid_eligible_pass(tmp_path):
    commit = "a" * 40
    cal = tmp_path / "cal.scored.jsonl"
    test = tmp_path / "test.scored.jsonl"
    out = tmp_path / "sensitivity.json"
    rows = [
        (f"c{i}", i % 2 == 0)
        for i in range(1000)
    ]
    _write_scored_rows(cal, rows)
    _write_scored_rows(test, [("t0", True), ("t1", False)])
    provenance = _analysis_binding(commit, cal, test)
    metrics = {
        "brier": 0.2,
        "nll": 0.6,
        "ece": 0.1,
        "ece_10": 0.1,
        "ece_20": 0.1,
        "aurc": 0.3,
        "risk_at_50_coverage": 0.2,
        "risk_at_80_coverage": 0.3,
    }
    improvement = {name: 0.01 for name in metrics}
    payload = {
        "protocol": "trajuq_isotonic_calibration_sensitivity_v1",
        "role": "supplementary_calibration_sensitivity",
        "status": "PASS",
        "provenance": provenance,
        "selective_risk_version": "tie_aware_uniform_tie_break_v2",
        "seed": 2027,
        "n_calibration": 1000,
        "n_test": 2,
        "eligibility": {
            "eligible": True,
            "min_calibration_samples_required": 1000,
            "min_class_count_required": 5,
            "cv_folds_requested": 5,
            "observed_class_counts": {"0": 500, "1": 500},
            "observed_min_class_count": 500,
        },
        "method": {
            "base_estimator": "standardized_logistic",
            "isotonic_cv": "stratified_out_of_fold",
            "ensemble": False,
            "test_labels_used_for_fit": False,
        },
        "effective_cv_folds": {
            "decision_entropy": 5,
            "trajectory_plus_decision": 5,
        },
        "decision_entropy": {
            "primary_logistic": metrics,
            "isotonic_sensitivity": metrics,
            "isotonic_minus_logistic_improvement": improvement,
        },
        "trajectory_plus_decision": {
            "primary_logistic": metrics,
            "isotonic_sensitivity": metrics,
            "isotonic_minus_logistic_improvement": improvement,
        },
    }
    out.write_text(json.dumps(payload), encoding="utf-8")
    cell = {
        "experiment_git_commit": commit,
        "calibration_sensitivity_contract": _isotonic_contract(),
        "inputs": {
            "calibration": {"path": str(cal), "sha256": _sha256(cal)},
            "test": {"path": str(test), "sha256": _sha256(test)},
        },
    }

    assert semantic_output_valid(cell, "calibration_sensitivity", out)
