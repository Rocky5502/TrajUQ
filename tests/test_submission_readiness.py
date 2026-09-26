import hashlib
import json

import yaml

import scripts.submission_readiness as submission_readiness
from scripts.render_results_tex import build_macros, render_tex
from scripts.submission_readiness import (
    preprimary_revalidation_argv,
    verify_downstream_provenance,
    verify_generated_results,
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
        "protocol": "trajuq_vs_repeated_macro_v1",
        "comparison": "single_trajectory_trajuq_vs_five_sample_baseline",
        "cell_set_complete": True,
        "expected_cells": 18,
        "observed_cells": 18,
        "macro": {
            "auroc": {"baseline_macro": 0.58, "candidate_macro": 0.61},
            "auprc": {"baseline_macro": 0.48, "candidate_macro": 0.51},
            "aurc": {"baseline_macro": 0.32, "candidate_macro": 0.29},
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
        "statistics": {
            "primary_endpoint": "hierarchical_macro_auroc_improvement",
            "rq1_claim_gate": {
                "macro_auroc_ci_low_gt_zero": True,
                "min_positive_model_families": 2,
            },
        }
    }

    payloads = {
        "primary": primary,
        "claim_gate": claim,
        "repeated": repeated,
        "adaptive": adaptive,
    }
    sources = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        sources[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    config = tmp_path / "primary.yaml"
    config.write_text(yaml.safe_dump(primary_cfg), encoding="utf-8")
    sources["primary_config"] = {
        "path": str(config),
        "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    return sources, payloads, primary_cfg


def write_valid_render_bundle(tmp_path):
    sources, payloads, primary_cfg = make_render_sources(tmp_path)
    macros = build_macros(
        payloads["primary"],
        payloads["claim_gate"],
        payloads["repeated"],
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
                "protocol": "trajuq_paper_results_render_v1",
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


def test_generated_results_hash_gate_passes(tmp_path):
    generated, manifest, _ = write_valid_render_bundle(tmp_path)

    result = verify_generated_results(manifest, generated)
    assert result["status"] == "PASS"


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
                "protocol": "trajuq_paper_results_render_v1",
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
                "protocol": "trajuq_paper_results_render_v1",
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
