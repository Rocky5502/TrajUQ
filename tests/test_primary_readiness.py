import hashlib
import json
from pathlib import Path

import yaml

from scripts.primary_readiness import evaluate_readiness


EXPERIMENT_COMMIT = "abc123frozen"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_cpu_pass(path: Path, commit: str = EXPERIMENT_COMMIT) -> None:
    write_json(
        path,
        {
            "status": "PASS",
            "git_commit": commit,
            "git_clean": True,
            "expected_steps": ["compileall", "pytest"],
            "completed_steps": ["compileall", "pytest"],
            "steps": [
                {"name": "compileall", "status": "PASS"},
                {"name": "pytest", "status": "PASS"},
            ],
        },
    )


def write_config(path: Path, models: tuple[str, ...]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "primary": [
                        f"configs/models/{model}.yaml"
                        for model in models
                    ]
                },
                "datasets": ["gsm8k"],
            }
        ),
        encoding="utf-8",
    )


def write_revision_ledger(
    path: Path,
    config: Path,
    models: tuple[str, ...],
    *,
    commit: str = EXPERIMENT_COMMIT,
) -> dict[str, str]:
    frozen_configs: dict[str, str] = {}
    frozen_hashes: dict[str, str] = {}
    model_rows: list[dict] = []
    for model in models:
        frozen = path.parent / f"{model}.frozen.yaml"
        frozen.write_text(
            yaml.safe_dump(
                {
                    "adapter": model,
                    "model_id": f"example/{model}",
                    "revision": f"rev-{model}",
                }
            ),
            encoding="utf-8",
        )
        frozen_configs[model] = str(frozen)
        frozen_hashes[model] = sha256_file(frozen)
        model_rows.append(
            {
                "adapter": model,
                "model_id": f"example/{model}",
                "resolved_sha": f"rev-{model}",
            }
        )

    write_json(
        path,
        {
            "protocol": "trajuq_revision_ledger_v2",
            "status": "PASS",
            "git_commit": commit,
            "primary_config": str(config),
            "primary_config_sha256": sha256_file(config),
            "dataset_manifest": "data/manifests/evaluation_splits.json",
            "dataset_manifest_sha256": "unused-in-unit-test",
            "models": model_rows,
            "datasets": [
                {
                    "name": "gsm8k",
                    "repo_id": "openai/gsm8k",
                    "resolved_sha": "dataset-rev",
                }
            ],
            "frozen_model_configs": frozen_configs,
            "frozen_model_config_sha256": frozen_hashes,
            "failures": [],
        },
    )
    return {model: f"rev-{model}" for model in models}


def write_parity(
    path: Path,
    model: str,
    revision: str,
    *,
    commit: str = EXPERIMENT_COMMIT,
) -> None:
    write_json(
        path,
        {
            "protocol": "trajuq_exact_parity_v3",
            "status": "PASS",
            "git_commit": commit,
            "require_token_ids": True,
            "require_provenance": True,
            "mismatch_count": 0,
            "provenance_mismatch_count": 0,
            "shared_records": 20,
            "run_provenance": {
                "dataset": "gsm8k",
                "model": f"example/{model}",
                "adapter": model,
                "seed": 2027,
                "model_revision": revision,
            },
        },
    )


def write_common_gates(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    quality = tmp_path / "quality.json"
    manual = tmp_path / "manual.json"
    math_audit = tmp_path / "math_audit.json"
    math_manual = tmp_path / "math_manual.json"
    cpu = tmp_path / "cpu.json"

    write_json(
        quality,
        {
            "status": "PASS",
            "raw_records": 50,
            "scored_records": 50,
            "model_revisions": ["rev-llada"],
        },
    )
    write_json(
        manual,
        {
            "status": "PASS",
            "reviewed_records": 10,
            "checks": {
                "trajectory_semantics": True,
                "scorer_audit": True,
                "parity_artifact_reviewed": True,
                "no_leakage_observed": True,
            },
        },
    )
    write_json(math_audit, {"status": "PASS", "failure_count": 0})
    write_json(
        math_manual,
        {
            "status": "PASS",
            "reviewed_records": 25,
            "checks": {
                "answer_extraction": True,
                "symbolic_equivalence": True,
                "no_manual_label_flips": True,
            },
        },
    )
    write_cpu_pass(cpu)
    return quality, manual, math_audit, math_manual, cpu


def evaluate(
    config: Path,
    quality: Path,
    manual: Path,
    parity: dict[str, str],
    math_audit: Path,
    math_manual: Path,
    cpu: Path,
    ledger: Path,
) -> dict:
    return evaluate_readiness(
        str(config),
        str(quality),
        str(manual),
        parity,
        str(math_audit),
        str(math_manual),
        str(cpu),
        str(ledger),
    )


def test_primary_readiness_passes_only_with_all_gates(tmp_path):
    models = ("llada", "sdar", "dream")
    config = tmp_path / "primary.yaml"
    ledger = tmp_path / "ledger.json"
    write_config(config, models)
    revisions = write_revision_ledger(ledger, config, models)
    quality, manual, math_audit, math_manual, cpu = write_common_gates(tmp_path)

    parity: dict[str, str] = {}
    for model in models:
        parity_path = tmp_path / f"{model}.json"
        write_parity(parity_path, model, revisions[model])
        parity[model] = str(parity_path)

    result = evaluate(
        config,
        quality,
        manual,
        parity,
        math_audit,
        math_manual,
        cpu,
        ledger,
    )
    assert result["status"] == "PASS"
    assert result["experiment_git_commit"] == EXPERIMENT_COMMIT
    assert not result["failures"]


def test_primary_readiness_fails_when_model_parity_missing(tmp_path):
    models = ("llada",)
    config = tmp_path / "primary.yaml"
    ledger = tmp_path / "ledger.json"
    write_config(config, models)
    write_revision_ledger(ledger, config, models)
    quality, manual, math_audit, math_manual, cpu = write_common_gates(tmp_path)

    result = evaluate(
        config,
        quality,
        manual,
        {},
        math_audit,
        math_manual,
        cpu,
        ledger,
    )
    assert result["status"] == "FAIL"
    assert any("parity" in item for item in result["failures"])


def test_primary_readiness_fails_without_math500_manual_gate(tmp_path):
    models = ("llada",)
    config = tmp_path / "primary.yaml"
    ledger = tmp_path / "ledger.json"
    write_config(config, models)
    revisions = write_revision_ledger(ledger, config, models)
    quality, manual, math_audit, math_manual, cpu = write_common_gates(tmp_path)

    write_json(
        math_manual,
        {
            "status": "FAIL",
            "reviewed_records": 0,
            "checks": {
                "answer_extraction": False,
                "symbolic_equivalence": False,
                "no_manual_label_flips": False,
            },
        },
    )
    parity_path = tmp_path / "llada.json"
    write_parity(parity_path, "llada", revisions["llada"])

    result = evaluate(
        config,
        quality,
        manual,
        {"llada": str(parity_path)},
        math_audit,
        math_manual,
        cpu,
        ledger,
    )
    assert result["status"] == "FAIL"
    assert any("MATH-500 manual" in item for item in result["failures"])


def test_primary_readiness_rejects_stale_cpu_and_parity_commit(tmp_path):
    models = ("llada",)
    config = tmp_path / "primary.yaml"
    ledger = tmp_path / "ledger.json"
    write_config(config, models)
    revisions = write_revision_ledger(ledger, config, models)
    quality, manual, math_audit, math_manual, cpu = write_common_gates(tmp_path)
    write_cpu_pass(cpu, commit="old-commit")

    parity_path = tmp_path / "llada.json"
    write_parity(
        parity_path,
        "llada",
        revisions["llada"],
        commit="old-commit",
    )

    result = evaluate(
        config,
        quality,
        manual,
        {"llada": str(parity_path)},
        math_audit,
        math_manual,
        cpu,
        ledger,
    )
    assert result["status"] == "FAIL"
    assert any("frozen experiment commit" in item for item in result["failures"])
    assert any("frozen revision/commit" in item for item in result["failures"])


def test_primary_readiness_rejects_parity_revision_not_in_ledger(tmp_path):
    models = ("llada",)
    config = tmp_path / "primary.yaml"
    ledger = tmp_path / "ledger.json"
    write_config(config, models)
    write_revision_ledger(ledger, config, models)
    quality, manual, math_audit, math_manual, cpu = write_common_gates(tmp_path)

    parity_path = tmp_path / "llada.json"
    write_parity(parity_path, "llada", "wrong-revision")

    result = evaluate(
        config,
        quality,
        manual,
        {"llada": str(parity_path)},
        math_audit,
        math_manual,
        cpu,
        ledger,
    )
    assert result["status"] == "FAIL"
    assert any("frozen revision/commit" in item for item in result["failures"])
