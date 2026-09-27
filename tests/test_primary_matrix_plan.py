from pathlib import Path

import yaml

from scripts.plan_primary_matrix import (
    generation_args,
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
