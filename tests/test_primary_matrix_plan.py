from pathlib import Path

import yaml

from trajuq.datasets import SPLIT_SELECTION_VERSION

from scripts.plan_primary_matrix import (
    generation_args,
    validate_prepared_split_identity,
    validate_readiness_source_bindings,
)


def test_generation_args_cover_all_primary_adapters():
    llada = {
        "adapter": "llada",
        "generation": {
            "steps": 128,
            "gen_length": 128,
            "block_length": 128,
            "temperature": 0.0,
            "cfg_scale": 0.0,
            "remasking": "low_confidence",
        },
    }
    dream = {
        "adapter": "dream",
        "generation": {
            "steps": 128,
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "alg": "entropy",
            "alg_temp": 0.1,
        },
    }
    sdar = {
        "adapter": "sdar",
        "generation": {
            "gen_length": 128,
            "block_length": 4,
            "denoising_steps": 4,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "remasking_strategy": "low_confidence_dynamic",
            "confidence_threshold": 0.85,
        },
    }

    assert "--llada-remasking" in generation_args(llada)
    assert "--dream-alg" in generation_args(dream)
    assert "--sdar-denoising-steps" in generation_args(sdar)
    assert "1.0" in generation_args(sdar)


def test_primary_config_is_three_by_six():
    cfg = yaml.safe_load(Path("configs/experiments/primary.yaml").read_text(encoding="utf-8"))
    assert len(cfg["models"]["primary"]) == 3
    assert len(cfg["datasets"]) == 6


def test_primary_planner_revalidates_split_and_revision_provenance():
    text = Path("scripts/plan_primary_matrix.py").read_text(encoding="utf-8")

    assert "audit_split_pair(cal, test, manifest)" in text
    assert "saved split audit is stale" in text
    assert "prepared split revision differs from revision ledger" in text
    assert '"revision_ledger_sha256"' in text
    assert "frozen model config hash differs from revision ledger" in text
    assert "frozen_model_config_sha256" in text
    assert "frozen model config path differs from revision ledger" in text
    assert '"split_manifest_sha256"' in text
    assert '"split_audit_sha256"' in text


def test_primary_planner_rejects_drift_in_readiness_bound_source(tmp_path):
    source = tmp_path / "parity.json"
    source.write_text('{"status":"PASS"}\n', encoding="utf-8")
    import hashlib

    readiness = {
        "source_bindings": [
            {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ]
    }
    assert validate_readiness_source_bindings(readiness) == []

    source.write_text('{"status":"FAIL"}\n', encoding="utf-8")
    failures = validate_readiness_source_bindings(readiness)
    assert any("bound source hash drifted" in item for item in failures)


def test_primary_planner_requires_readiness_source_manifest():
    assert validate_readiness_source_bindings({}) == [
        "primary readiness has no source-binding manifest"
    ]


def test_prepared_split_identity_matches_frozen_dataset_policy():
    spec = {
        "hf_id": "openai/gsm8k",
        "config": "main",
        "strategy": "official",
        "calibration_split": "train",
        "test_split": "test",
        "source_split": None,
        "calibration_fraction": None,
        "split_seed": None,
        "pre_generation_cap": {
            "method": SPLIT_SELECTION_VERSION,
            "seed": "2027",
            "calibration_max": 500,
            "test_max": 500,
        },
    }
    meta = {
        **spec,
        "selection_version": SPLIT_SELECTION_VERSION,
        "source_counts": {
            "source_total": 5,
            "calibration_before_cap": 2,
            "test_before_cap": 3,
        },
        "n_calibration": 2,
        "n_test": 3,
        "source_git_commit": "frozen-commit",
        "source_git_clean": True,
        "requested_revision": "dataset-sha",
    }
    ledger = {
        "repo_id": "openai/gsm8k",
        "resolved_sha": "dataset-sha",
    }

    assert validate_prepared_split_identity(
        "gsm8k", meta, spec, ledger, "frozen-commit"
    ) == []


def test_prepared_split_identity_rejects_policy_drift():
    spec = {
        "hf_id": "Idavidrein/gpqa",
        "config": "gpqa_diamond",
        "strategy": "internal_holdout",
        "calibration_split": None,
        "test_split": None,
        "source_split": "train",
        "calibration_fraction": 0.30,
        "split_seed": "2027",
        "pre_generation_cap": {
            "method": SPLIT_SELECTION_VERSION,
            "seed": "2027",
            "calibration_max": None,
            "test_max": None,
        },
    }
    meta = {
        **spec,
        "selection_version": SPLIT_SELECTION_VERSION,
        "source_counts": {
            "source_total": 5,
            "calibration_before_cap": 2,
            "test_before_cap": 3,
        },
        "n_calibration": 2,
        "n_test": 3,
        "source_git_commit": "frozen-commit",
        "source_git_clean": True,
        "requested_revision": "dataset-sha",
        "calibration_fraction": 0.20,
        "split_seed": "other-seed",
    }
    ledger = {
        "repo_id": "Idavidrein/gpqa",
        "resolved_sha": "dataset-sha",
    }

    failures = validate_prepared_split_identity(
        "gpqa_diamond", meta, spec, ledger, "frozen-commit"
    )
    assert any("calibration_fraction differs" in item for item in failures)
    assert any("split_seed differs" in item for item in failures)


def test_prepared_split_identity_rejects_repo_and_revision_drift():
    spec = {
        "hf_id": "openai/gsm8k",
        "config": "main",
        "strategy": "official",
        "calibration_split": "train",
        "test_split": "test",
        "source_split": None,
        "calibration_fraction": None,
        "split_seed": None,
        "pre_generation_cap": {
            "method": SPLIT_SELECTION_VERSION,
            "seed": "2027",
            "calibration_max": 500,
            "test_max": 500,
        },
    }
    meta = {
        **spec,
        "selection_version": SPLIT_SELECTION_VERSION,
        "source_counts": {
            "source_total": 5,
            "calibration_before_cap": 2,
            "test_before_cap": 3,
        },
        "n_calibration": 2,
        "n_test": 3,
        "source_git_commit": "frozen-commit",
        "source_git_clean": True,
        "hf_id": "other/gsm8k",
        "requested_revision": "wrong-sha",
    }
    ledger = {
        "repo_id": "openai/gsm8k",
        "resolved_sha": "dataset-sha",
    }

    failures = validate_prepared_split_identity(
        "gsm8k", meta, spec, ledger, "frozen-commit"
    )
    assert any("repo_id differs" in item for item in failures)
    assert any("revision differs" in item for item in failures)


def test_prepared_split_identity_rejects_other_source_commit():
    spec = {
        "hf_id": "openai/gsm8k",
        "config": "main",
        "strategy": "official",
        "calibration_split": "train",
        "test_split": "test",
        "source_split": None,
        "calibration_fraction": None,
        "split_seed": None,
        "pre_generation_cap": {
            "method": SPLIT_SELECTION_VERSION,
            "seed": "2027",
            "calibration_max": 500,
            "test_max": 500,
        },
    }
    meta = {
        **spec,
        "selection_version": SPLIT_SELECTION_VERSION,
        "source_counts": {
            "source_total": 5,
            "calibration_before_cap": 2,
            "test_before_cap": 3,
        },
        "n_calibration": 2,
        "n_test": 3,
        "source_git_commit": "old-commit",
        "source_git_clean": True,
        "requested_revision": "dataset-sha",
    }
    ledger = {
        "repo_id": "openai/gsm8k",
        "resolved_sha": "dataset-sha",
    }

    failures = validate_prepared_split_identity(
        "gsm8k", meta, spec, ledger, "frozen-commit"
    )
    assert any("source commit differs" in item for item in failures)



def test_prepared_split_identity_rejects_cap_policy_drift():
    spec = {
        "hf_id": "openai/gsm8k",
        "config": "main",
        "strategy": "official",
        "calibration_split": "train",
        "test_split": "test",
        "source_split": None,
        "calibration_fraction": None,
        "split_seed": None,
        "pre_generation_cap": {
            "method": SPLIT_SELECTION_VERSION,
            "seed": "2027",
            "calibration_max": 500,
            "test_max": 500,
        },
    }
    meta = {
        **spec,
        "pre_generation_cap": {
            **spec["pre_generation_cap"],
            "test_max": 1000,
        },
        "selection_version": SPLIT_SELECTION_VERSION,
        "source_counts": {
            "source_total": 1600,
            "calibration_before_cap": 800,
            "test_before_cap": 800,
        },
        "n_calibration": 500,
        "n_test": 500,
        "source_git_commit": "frozen-commit",
        "source_git_clean": True,
        "requested_revision": "dataset-sha",
    }
    ledger = {
        "repo_id": "openai/gsm8k",
        "resolved_sha": "dataset-sha",
    }

    failures = validate_prepared_split_identity(
        "gsm8k", meta, spec, ledger, "frozen-commit"
    )

    assert any("pre_generation_cap differs" in item for item in failures)
