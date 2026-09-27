import hashlib
import json
import zipfile

import yaml

import scripts.submission_readiness as submission_readiness
from scripts.render_results_tex import (
    ABLATION_SUFFIXES,
    PRIOR_SIGNAL_SUFFIXES,
    build_macros,
    render_tex,
)
from scripts.submission_readiness import (
    prerequisite_gate_failures,
    preprimary_revalidation_argv,
    verify_downstream_provenance,
    verify_generated_results,
    verify_overleaf_package,
    verify_result_figures,
    verify_submission_checkout,
)


def make_render_sources(tmp_path):
    primary = {
        "protocol": "hierarchical_within_cell_macro_bootstrap_v1",
        "cell_set_complete": True,
        "expected_cell_count": 18,
        "observed_cell_count": 18,
        "macro": {
            "auroc": {
                "baseline_macro": 0.60,
                "candidate_macro": 0.61,
                "improvement_macro": 0.01,
                "improvement_ci_low": -0.01,
                "improvement_ci_high": 0.03,
                "positive_model_families": 2,
                "n_model_families": 3,
            },
            "auprc": {"baseline_macro": 0.50, "candidate_macro": 0.51},
            "brier": {"baseline_macro": 0.20, "candidate_macro": 0.19},
            "ece15": {"baseline_macro": 0.10, "candidate_macro": 0.09},
            "aurc": {"baseline_macro": 0.30, "candidate_macro": 0.29},
        },
    }
    claim = {
        "protocol": "trajuq_rq1_claim_gate_v1",
        "primary_endpoint": "hierarchical_macro_auroc_improvement",
        "decision": "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM",
        "claim_supported": False,
        "frozen_rules": {
            "macro_auroc_ci_low_gt_zero": True,
            "min_positive_model_families": 2,
        },
        "observed": {
            "macro_auroc_improvement": 0.01,
            "macro_auroc_ci_low": -0.01,
            "macro_auroc_ci_high": 0.03,
            "positive_model_families": 2,
            "n_model_families": 3,
        },
        "conditions": {
            "complete_frozen_cell_set": True,
            "macro_auroc_improvement_positive": True,
            "macro_auroc_ci_low_gt_zero": False,
            "positive_model_families_at_least_minimum": True,
            "three_primary_model_families_present": True,
        },
    }
    repeated = {
        "protocol": "trajuq_vs_repeated_macro_v2",
        "comparison": "single_trajectory_trajuq_vs_five_sample_baseline",
        "cell_set_complete": True,
        "expected_cells": 12,
        "observed_cells": 12,
        "eligible_datasets": [
            "gsm8k",
            "math500",
            "gpqa_diamond",
            "mmlu_pro",
        ],
        "excluded_datasets": {
            "arc_challenge": "frozen compute-bounded exclusion",
            "truthfulqa": "frozen compute-bounded exclusion",
        },
        "auxiliary_compute": {
            "aggregation": "equal_cell_weight_macro",
            "expected_cells": 12,
            "observed_cells_with_complete_test_wall_time": 12,
            "complete": True,
            "sample_count": 5,
            "mean_test_auxiliary_wall_time_seconds_per_example": 12.34,
            "mean_test_p90_auxiliary_wall_time_seconds_per_example": 15.67,
        },
        "macro": {
            "auroc": {"baseline_macro": 0.58, "candidate_macro": 0.61},
            "auprc": {"baseline_macro": 0.48, "candidate_macro": 0.51},
            "aurc": {"baseline_macro": 0.32, "candidate_macro": 0.29},
        },
    }
    ablation = {
        "protocol": "trajuq_feature_ablation_macro_v1",
        "baseline": "decision_only",
        "cell_set_complete": True,
        "expected_cells": 18,
        "observed_cells": 18,
        "groups": {
            name: {
                "metrics": {
                    metric: {
                        "baseline_macro": 0.60,
                        "candidate_macro": 0.61,
                        "improvement_macro": 0.01,
                        "improvement_ci_low": -0.01,
                        "improvement_ci_high": 0.03,
                    }
                    for metric in ("auroc", "auprc", "aurc")
                }
            }
            for name in ABLATION_SUFFIXES
        },
    }
    prior_signals = {
        "protocol": "trajuq_prior_aligned_signal_macro_v1",
        "role": "secondary_prior_aligned_signal_class_diagnostic",
        "cell_set_complete": True,
        "expected_cells": 18,
        "observed_cells": 18,
        "macro": {
            name: {
                metric: {
                    "baseline_macro": 0.59,
                    "candidate_macro": 0.61,
                    "improvement_macro": 0.02,
                    "improvement_ci_low": -0.01,
                    "improvement_ci_high": 0.04,
                }
                for metric in ("auroc", "auprc", "aurc")
            }
            for name in PRIOR_SIGNAL_SUFFIXES
        },
    }
    sensitivity_labels = [
        f"{adapter}:{dataset}"
        for adapter in ("llada", "sdar", "dream")
        for dataset in (
            "gsm8k",
            "math500",
            "gpqa_diamond",
            "arc_challenge",
            "mmlu_pro",
            "truthfulqa",
        )
    ]
    sensitivity_eligible = sensitivity_labels[:6]
    sensitivity_skipped = sensitivity_labels[6:]
    sensitivity_metric = {
        "primary_logistic": 0.20,
        "isotonic_sensitivity": 0.18,
        "isotonic_minus_logistic_improvement": 0.02,
    }
    calibration_sensitivity = {
        "protocol": "trajuq_isotonic_calibration_sensitivity_macro_v1",
        "role": "supplementary_calibration_sensitivity",
        "cell_set_complete": True,
        "expected_cells": 18,
        "observed_cells": 18,
        "eligibility_policy": {
            "role": "supplementary_calibration_sensitivity",
            "base_estimator": "standardized_logistic",
            "cv_folds": 5,
            "ensemble": False,
            "min_calibration_samples": 1000,
            "min_class_count": 5,
        },
        "eligible_cell_count": len(sensitivity_eligible),
        "skipped_cell_count": len(sensitivity_skipped),
        "eligible_cells": sensitivity_eligible,
        "skipped_cells": sensitivity_skipped,
        "aggregation": (
            "equal_weight_over_pre_registered_eligible_cells_only; all frozen "
            "ineligible cells remain explicit skips"
        ),
        "delta_direction": (
            "positive means isotonic improves a lower-is-better metric "
            "relative to the primary logistic calibrator"
        ),
        "metrics": [
            "brier",
            "nll",
            "ece",
            "ece_10",
            "ece_20",
            "aurc",
            "risk_at_50_coverage",
            "risk_at_80_coverage",
        ],
        "per_cell": {
            label: {
                "status": (
                    "PASS"
                    if label in sensitivity_eligible
                    else "SKIP_INSUFFICIENT_CALIBRATION"
                ),
                "eligible": label in sensitivity_eligible,
            }
            for label in sensitivity_labels
        },
        "macro_over_eligible_cells": {
            view: {
                metric: dict(sensitivity_metric)
                for metric in (
                    "brier",
                    "nll",
                    "ece",
                    "ece_10",
                    "ece_20",
                    "aurc",
                    "risk_at_50_coverage",
                    "risk_at_80_coverage",
                )
            }
            for view in (
                "decision_entropy",
                "trajectory_plus_decision",
            )
        },
    }
    adaptive = {
        "protocol": "trajuq_adaptive_macro_v1",
        "execution_semantics": "selective_second_pass_rerun",
        "fraction_key": "cap_050",
        "cell_set_complete": True,
        "expected_cells": 18,
        "observed_cells": 18,
        "is_primary_fraction": True,
        "max_refine_fraction": 0.50,
        "macro": {
            "adaptive_accuracy": 0.70,
            "base_accuracy": 0.66,
            "refined_accuracy": 0.72,
            "random_expected_accuracy": 0.68,
            "delta_vs_random": 0.02,
            "nominal_cost_ratio_to_base": 1.50,
            "refined_cost_ratio_to_base": 4.0,
        },
    }
    primary_cfg = {
        "models": {
            "primary": [
                "configs/models/llada.yaml",
                "configs/models/sdar.yaml",
                "configs/models/dream.yaml",
            ]
        },
        "baselines": {
            "expensive": {
                "repeated_sampling": {
                    "eligible_datasets": [
                        "gsm8k",
                        "math500",
                        "gpqa_diamond",
                        "mmlu_pro",
                    ]
                }
            }
        },
        "statistics": {
            "primary_endpoint": "hierarchical_macro_auroc_improvement",
            "rq1_claim_gate": {
                "macro_auroc_ci_low_gt_zero": True,
                "min_positive_model_families": 2,
            },
        }
    }

    commit = "a" * 40
    config = tmp_path / "primary.yaml"
    config.write_text(yaml.safe_dump(primary_cfg), encoding="utf-8")
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()

    primary_prediction = tmp_path / "primary_prediction.jsonl"
    ablation_prediction = tmp_path / "ablation_prediction.jsonl"
    prior_prediction = tmp_path / "prior_prediction.jsonl"
    repeated_source = tmp_path / "repeated_plan.json"
    adaptive_source = tmp_path / "adaptive_plan.json"
    calibration_sensitivity_source = tmp_path / "calibration_sensitivity_cell.json"
    for path in (
        primary_prediction,
        ablation_prediction,
        prior_prediction,
        repeated_source,
        adaptive_source,
        calibration_sensitivity_source,
    ):
        path.write_text('{"example_id":"x"}\n', encoding="utf-8")

    primary["provenance"] = {
        "protocol": "trajuq_analysis_source_binding_v1",
        "git_commit": commit,
        "git_clean": True,
        "sources": {
            "primary_config": {
                "path": str(config),
                "sha256": config_sha,
            },
            "prediction:llada_gsm8k": {
                "path": str(primary_prediction),
                "sha256": hashlib.sha256(
                    primary_prediction.read_bytes()
                ).hexdigest(),
            },
        },
    }
    ablation["provenance"] = {
        "protocol": "trajuq_analysis_source_binding_v1",
        "git_commit": commit,
        "git_clean": True,
        "sources": {
            "prediction:llada:gsm8k": {
                "path": str(ablation_prediction),
                "sha256": hashlib.sha256(
                    ablation_prediction.read_bytes()
                ).hexdigest(),
            }
        },
    }
    prior_signals["provenance"] = {
        "protocol": "trajuq_analysis_source_binding_v1",
        "git_commit": commit,
        "git_clean": True,
        "sources": {
            "prior_signal_prediction:llada:gsm8k": {
                "path": str(prior_prediction),
                "sha256": hashlib.sha256(
                    prior_prediction.read_bytes()
                ).hexdigest(),
            }
        },
    }
    repeated["provenance"] = {
        "protocol": "trajuq_analysis_source_binding_v1",
        "git_commit": commit,
        "git_clean": True,
        "sources": {
            "plan": {
                "path": str(repeated_source),
                "sha256": hashlib.sha256(
                    repeated_source.read_bytes()
                ).hexdigest(),
            }
        },
    }
    calibration_sensitivity["provenance"] = {
        "protocol": "trajuq_analysis_source_binding_v1",
        "git_commit": commit,
        "git_clean": True,
        "sources": {
            "primary_config": {
                "path": str(config),
                "sha256": config_sha,
            },
            **{
                f"cell:{label}": {
                    "path": str(calibration_sensitivity_source),
                    "sha256": hashlib.sha256(
                        calibration_sensitivity_source.read_bytes()
                    ).hexdigest(),
                }
                for label in sensitivity_labels
            },
        },
    }
    adaptive["provenance"] = {
        "protocol": "trajuq_analysis_source_binding_v1",
        "git_commit": commit,
        "git_clean": True,
        "sources": {
            "plan": {
                "path": str(adaptive_source),
                "sha256": hashlib.sha256(
                    adaptive_source.read_bytes()
                ).hexdigest(),
            }
        },
    }

    primary_path = tmp_path / "primary.json"
    primary_path.write_text(json.dumps(primary), encoding="utf-8")
    claim["provenance"] = {
        "protocol": "trajuq_analysis_source_binding_v1",
        "git_commit": commit,
        "git_clean": True,
        "sources": {
            "macro_primary": {
                "path": str(primary_path),
                "sha256": hashlib.sha256(
                    primary_path.read_bytes()
                ).hexdigest(),
            },
            "primary_config": {
                "path": str(config),
                "sha256": config_sha,
            },
        },
    }

    payloads = {
        "primary": primary,
        "claim_gate": claim,
        "repeated": repeated,
        "ablation": ablation,
        "prior_signals": prior_signals,
        "calibration_sensitivity": calibration_sensitivity,
        "adaptive": adaptive,
    }
    sources = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        if name == "primary":
            path = primary_path
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")
        sources[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    sources["primary_config"] = {
        "path": str(config),
        "sha256": config_sha,
    }
    return sources, payloads, primary_cfg


def write_valid_render_bundle(tmp_path):
    sources, payloads, primary_cfg = make_render_sources(tmp_path)
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
    render_sources = {
        name: (source["path"], source["sha256"])
        for name, source in sources.items()
    }
    generated = tmp_path / "generated_results.tex"
    generated.write_text(
        render_tex(macros, render_sources),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "trajuq_paper_results_render_v2",
                "output_sha256": hashlib.sha256(
                    generated.read_bytes()
                ).hexdigest(),
                "sources": sources,
                "macro_count": len(macros),
                "rq1_claim_decision": payloads["claim_gate"]["decision"],
            }
        ),
        encoding="utf-8",
    )
    return generated, manifest, sources


def write_overleaf_zip(path, *, extra_pdf: bool = False, ready: bool = True):
    payloads = {
        "main.tex": b"main\n",
        "appendix.tex": b"appendix\n",
        "references.bib": b"bib\n",
        "generated_results.tex": b"results\n",
        "acl.sty": b"style\n",
        "acl_natbib.bst": b"bst\n",
        "PAPER_STATUS.md": b"status\n",
        "figures/generated/primary_auroc_improvement_matrix.pdf": (
            b"%PDF-1.4\nprimary\n"
        ),
        "figures/generated/family_macro_auroc_improvement.pdf": (
            b"%PDF-1.4\nfamily\n"
        ),
        "figures/generated/adaptive_accuracy_cost_tradeoff.pdf": (
            b"%PDF-1.4\nadaptive\n"
        ),
    }
    if extra_pdf:
        payloads["figures/generated/stale.pdf"] = b"%PDF-1.4\nstale\n"

    manifest = {
        "protocol": "trajuq_overleaf_package_v1",
        "anonymous": True,
        "submission_ready": ready,
        "acl_style_commit": submission_readiness.ACL_STYLE_COMMIT,
        "files": [
            {
                "path": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
            for name, data in sorted(payloads.items())
        ],
    }
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in payloads.items():
            zf.writestr(name, data)
        zf.writestr(
            "MANIFEST.json",
            json.dumps(manifest).encode("utf-8"),
        )


def test_overleaf_archive_gate_passes_for_exact_submission_bundle(tmp_path):
    archive = tmp_path / "overleaf.zip"
    write_overleaf_zip(archive)
    result = verify_overleaf_package(archive)
    assert result["status"] == "PASS"
    assert result["sha256"] == hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()


def test_overleaf_archive_gate_rejects_extra_generated_pdf(tmp_path):
    archive = tmp_path / "overleaf.zip"
    write_overleaf_zip(archive, extra_pdf=True)
    result = verify_overleaf_package(archive)
    assert result["status"] == "FAIL"
    assert any(
        "generated-PDF set differs" in item
        for item in result["failures"]
    )


def test_overleaf_archive_gate_requires_submission_ready_marker(tmp_path):
    archive = tmp_path / "overleaf.zip"
    write_overleaf_zip(archive, ready=False)
    result = verify_overleaf_package(archive)
    assert result["status"] == "FAIL"
    assert any(
        "not submission-ready" in item
        for item in result["failures"]
    )


def test_generated_results_hash_gate_passes(tmp_path):
    generated, manifest, _ = write_valid_render_bundle(tmp_path)

    result = verify_generated_results(manifest, generated)
    assert result["status"] == "PASS"


def test_generated_results_gate_rejects_upstream_provenance_drift(tmp_path):
    generated, manifest, _ = write_valid_render_bundle(tmp_path)
    upstream = tmp_path / "primary_prediction.jsonl"
    upstream.write_text(
        '{"example_id":"x","changed":true}\n',
        encoding="utf-8",
    )

    result = verify_generated_results(manifest, generated)
    assert result["status"] == "FAIL"
    assert any(
        "provenance source hash changed" in item
        for item in result["failures"]
    )


def test_generated_results_hash_gate_rejects_changed_source(tmp_path):
    generated, manifest, _ = write_valid_render_bundle(tmp_path)
    source = tmp_path / "primary.json"
    source.write_text('{"ok": false}', encoding="utf-8")

    result = verify_generated_results(manifest, generated)
    assert result["status"] == "FAIL"
    assert any("source hash changed" in item for item in result["failures"])


def test_generated_results_gate_rejects_joint_tex_manifest_tampering(tmp_path):
    generated, manifest, _ = write_valid_render_bundle(tmp_path)
    generated.write_text(
        generated.read_text(encoding="utf-8").replace("0.610", "0.999"),
        encoding="utf-8",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["output_sha256"] = hashlib.sha256(
        generated.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_generated_results(manifest, generated)
    assert result["status"] == "FAIL"
    assert any(
        "differs from deterministic re-render" in item
        for item in result["failures"]
    )


def test_generated_results_gate_rejects_pre_results_placeholder(tmp_path):
    generated = tmp_path / "generated_results.tex"
    generated.write_text(
        "% PRE-RESULTS FALLBACK\n\\newcommand{\\X}{\\tbd}\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "trajuq_paper_results_render_v2",
                "output_sha256": hashlib.sha256(
                    generated.read_bytes()
                ).hexdigest(),
                "sources": {},
            }
        ),
        encoding="utf-8",
    )
    result = verify_generated_results(manifest, generated)
    assert result["status"] == "FAIL"


def test_downstream_provenance_revalidation_uses_historical_mode(
    tmp_path,
    monkeypatch,
):
    paths = []
    for name in ("primary", "analysis", "repeated", "adaptive"):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "protocol": f"{name}_protocol",
                    "experiment_git_commit": "a" * 40,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    calls = []

    def validator(payload, *, require_current_commit=True):
        calls.append(
            {
                "protocol": payload["protocol"],
                "require_current_commit": require_current_commit,
            }
        )
        return []

    monkeypatch.setattr(
        submission_readiness,
        "validate_primary_provenance",
        validator,
    )
    monkeypatch.setattr(
        submission_readiness,
        "validate_analysis_provenance",
        validator,
    )
    monkeypatch.setattr(
        submission_readiness,
        "validate_repeated_provenance",
        validator,
    )
    monkeypatch.setattr(
        submission_readiness,
        "validate_adaptive_provenance",
        validator,
    )

    result = verify_downstream_provenance(*paths)
    assert result["status"] == "PASS"
    assert len(calls) == 4
    assert all(
        call["require_current_commit"] is False
        for call in calls
    )


def test_downstream_provenance_revalidation_surfaces_drift(
    tmp_path,
    monkeypatch,
):
    paths = []
    for name in ("primary", "analysis", "repeated", "adaptive"):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "protocol": f"{name}_protocol",
                    "experiment_git_commit": "b" * 40,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    monkeypatch.setattr(
        submission_readiness,
        "validate_primary_provenance",
        lambda payload, *, require_current_commit=True: [],
    )
    monkeypatch.setattr(
        submission_readiness,
        "validate_analysis_provenance",
        lambda payload, *, require_current_commit=True: [],
    )
    monkeypatch.setattr(
        submission_readiness,
        "validate_repeated_provenance",
        lambda payload, *, require_current_commit=True: [
            "TrajUQ predictions hash drifted"
        ],
    )
    monkeypatch.setattr(
        submission_readiness,
        "validate_adaptive_provenance",
        lambda payload, *, require_current_commit=True: [],
    )

    result = verify_downstream_provenance(*paths)
    assert result["status"] == "FAIL"
    assert any(
        "repeated: TrajUQ predictions hash drifted" in failure
        for failure in result["failures"]
    )


def test_submission_revalidation_uses_historical_mode_and_separate_output():
    argv = preprimary_revalidation_argv()
    assert "scripts/primary_readiness.py" in argv
    assert "--historical-provenance-only" in argv
    assert "results/submission_preprimary_revalidation.json" in argv
    assert "results/primary_readiness.json" not in argv


def test_submission_checkout_allows_later_clean_commit(monkeypatch):
    def fake_git_text(*args):
        if args == ("rev-parse", "HEAD"):
            return "submission-commit"
        if args == ("status", "--porcelain"):
            return ""
        return None

    monkeypatch.setattr(submission_readiness, "git_text", fake_git_text)
    result = verify_submission_checkout()
    assert result["status"] == "PASS"
    assert result["git_commit"] == "submission-commit"
    assert result["git_clean"] is True


def test_submission_checkout_rejects_dirty_worktree(monkeypatch):
    def fake_git_text(*args):
        if args == ("rev-parse", "HEAD"):
            return "submission-commit"
        if args == ("status", "--porcelain"):
            return " M paper/main.tex"
        return None

    monkeypatch.setattr(submission_readiness, "git_text", fake_git_text)
    result = verify_submission_checkout()
    assert result["status"] == "FAIL"
    assert result["git_clean"] is False
    assert any("worktree is dirty" in item for item in result["failures"])


def test_generated_results_gate_rejects_missing_required_sources(tmp_path):
    generated = tmp_path / "generated_results.tex"
    generated.write_text(
        "\\newcommand{\\PrimaryTrajAUROC}{0.700}\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "trajuq_paper_results_render_v2",
                "output_sha256": hashlib.sha256(
                    generated.read_bytes()
                ).hexdigest(),
                "sources": {},
                "macro_count": 1,
                "rq1_claim_decision": "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM",
            }
        ),
        encoding="utf-8",
    )

    result = verify_generated_results(manifest, generated)
    assert result["status"] == "FAIL"
    assert any("missing required sources" in item for item in result["failures"])


def test_submission_prerequisites_exclude_own_output_gate():
    snapshot = {
        "execution_gates": [
            {
                "id": "primary",
                "required": True,
                "state": "PASS",
            },
            {
                "id": "submission_readiness",
                "required": True,
                "state": "PENDING",
            },
        ]
    }
    assert prerequisite_gate_failures(snapshot) == []


def test_submission_prerequisites_still_block_other_required_gate():
    snapshot = {
        "execution_gates": [
            {
                "id": "primary",
                "required": True,
                "state": "PENDING",
            },
            {
                "id": "submission_readiness",
                "required": True,
                "state": "PENDING",
            },
        ]
    }
    failures = prerequisite_gate_failures(snapshot)
    assert [item["id"] for item in failures] == ["primary"]


def test_result_figure_manifest_hash_gate_passes(tmp_path):
    source_paths = {}
    for name in ("primary", "primary_config", "cap_025", "cap_050", "cap_075"):
        path = tmp_path / f"{name}.json"
        path.write_text('{"ok":true}\n', encoding="utf-8")
        source_paths[name] = path

    outputs = {}
    for name in (
        "primary_auroc_matrix",
        "family_macro_auroc",
        "adaptive_tradeoff",
    ):
        path = tmp_path / f"{name}.pdf"
        path.write_bytes(b"%PDF-1.4\nsynthetic-test\n")
        outputs[name] = path

    manifest = tmp_path / "result_figure_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "trajuq_paper_result_figures_v2",
                "expected_cells": 18,
                "sources": {
                    name: {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in source_paths.items()
                },
                "outputs": {
                    name: {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                    }
                    for name, path in outputs.items()
                },
            }
        ),
        encoding="utf-8",
    )

    result = verify_result_figures(manifest)
    assert result["status"] == "PASS"


def test_result_figure_manifest_rejects_pdf_tampering(tmp_path):
    source_paths = {}
    for name in ("primary", "primary_config", "cap_025", "cap_050", "cap_075"):
        path = tmp_path / f"{name}.json"
        path.write_text('{"ok":true}\n', encoding="utf-8")
        source_paths[name] = path

    outputs = {}
    for name in (
        "primary_auroc_matrix",
        "family_macro_auroc",
        "adaptive_tradeoff",
    ):
        path = tmp_path / f"{name}.pdf"
        path.write_bytes(b"%PDF-1.4\nsynthetic-test\n")
        outputs[name] = path

    manifest = tmp_path / "result_figure_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "trajuq_paper_result_figures_v2",
                "expected_cells": 18,
                "sources": {
                    name: {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for name, path in source_paths.items()
                },
                "outputs": {
                    name: {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                    }
                    for name, path in outputs.items()
                },
            }
        ),
        encoding="utf-8",
    )
    outputs["adaptive_tradeoff"].write_bytes(b"%PDF-1.4\ntampered\n")

    result = verify_result_figures(manifest)
    assert result["status"] == "FAIL"
    assert any(
        "result figure hash changed: adaptive_tradeoff" in item
        for item in result["failures"]
    )


def test_result_figure_manifest_rejects_malformed_entries(tmp_path):
    manifest = tmp_path / "result_figure_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "trajuq_paper_result_figures_v2",
                "expected_cells": 18,
                "sources": {
                    "primary": "not-an-object",
                    "primary_config": {},
                    "cap_025": {},
                    "cap_050": {},
                    "cap_075": {},
                },
                "outputs": {
                    "primary_auroc_matrix": "not-an-object",
                    "family_macro_auroc": {},
                    "adaptive_tradeoff": {},
                },
            }
        ),
        encoding="utf-8",
    )

    result = verify_result_figures(manifest)
    assert result["status"] == "FAIL"
    assert any(
        "source entry is invalid: primary" in item
        for item in result["failures"]
    )
    assert any(
        "output entry is invalid: primary_auroc_matrix" in item
        for item in result["failures"]
    )


def test_generated_results_gate_binds_exact_rq2_rq3_plans(tmp_path):
    generated, manifest, _ = write_valid_render_bundle(tmp_path)
    repeated_plan = tmp_path / "repeated_plan.json"
    adaptive_plan = tmp_path / "adaptive_plan.json"

    result = verify_generated_results(
        manifest,
        generated,
        repeated_plan,
        adaptive_plan,
    )
    assert result["status"] == "PASS"

    wrong_repeated = tmp_path / "wrong_repeated_plan.json"
    wrong_repeated.write_bytes(repeated_plan.read_bytes())
    result = verify_generated_results(
        manifest,
        generated,
        wrong_repeated,
        adaptive_plan,
    )
    assert result["status"] == "FAIL"
    assert any(
        "repeated macro provenance binds a different frozen plan" in item
        for item in result["failures"]
    )


def test_result_figure_gate_binds_exact_adaptive_plan(tmp_path):
    sources, payloads, _ = make_render_sources(tmp_path)
    primary_path = tmp_path / "primary.json"
    adaptive_plan = tmp_path / "adaptive_plan.json"

    cap_paths = {}
    for key, fraction, is_primary in (
        ("cap_025", 0.25, False),
        ("cap_050", 0.50, True),
        ("cap_075", 0.75, False),
    ):
        payload = json.loads(json.dumps(payloads["adaptive"]))
        payload["fraction_key"] = key
        payload["max_refine_fraction"] = fraction
        payload["is_primary_fraction"] = is_primary
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        cap_paths[key] = path

    output_paths = {}
    for name in (
        "primary_auroc_matrix",
        "family_macro_auroc",
        "adaptive_tradeoff",
    ):
        path = tmp_path / f"{name}.pdf"
        path.write_bytes(b"%PDF-1.4\nsynthetic-test\n")
        output_paths[name] = path

    source_paths = {
        "primary": primary_path,
        "primary_config": tmp_path / "primary.yaml",
        **cap_paths,
    }
    manifest = tmp_path / "strict_result_figure_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol": "trajuq_paper_result_figures_v2",
                "expected_cells": 18,
                "sources": {
                    name: {
                        "path": str(path),
                        "sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                    }
                    for name, path in source_paths.items()
                },
                "outputs": {
                    name: {
                        "path": str(path),
                        "sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                        "bytes": path.stat().st_size,
                    }
                    for name, path in output_paths.items()
                },
            }
        ),
        encoding="utf-8",
    )

    result = verify_result_figures(manifest, adaptive_plan)
    assert result["status"] == "PASS"

    wrong_adaptive = tmp_path / "wrong_adaptive_plan.json"
    wrong_adaptive.write_bytes(adaptive_plan.read_bytes())
    result = verify_result_figures(manifest, wrong_adaptive)
    assert result["status"] == "FAIL"
    assert any(
        "provenance binds a different adaptive plan" in item
        for item in result["failures"]
    )
