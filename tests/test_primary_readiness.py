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


def build_case(
    tmp_path: Path,
    models: tuple[str, ...] = ("llada", "sdar", "dream"),
) -> dict:
    config = tmp_path / "primary.yaml"
    ledger = tmp_path / "ledger.json"
    cpu = tmp_path / "cpu.json"
    quality = tmp_path / "quality.json"
    manual = tmp_path / "manual.json"
    math_audit = tmp_path / "math_audit.json"
    math_manual = tmp_path / "math_manual.json"

    write_config(config, models)
    revisions = write_revision_ledger(ledger, config, models)
    write_cpu_pass(cpu)

    parity: dict[str, str] = {}
    for model in models:
        parity_path = tmp_path / f"{model}_parity.json"
        write_parity(parity_path, model, revisions[model])
        parity[model] = str(parity_path)

    raw = tmp_path / "pilot.raw.jsonl"
    scored = tmp_path / "pilot.scored.jsonl"
    raw.write_text('{"dummy":"raw"}\n', encoding="utf-8")
    scored.write_text('{"dummy":"scored"}\n', encoding="utf-8")
    write_json(
        quality,
        {
            "protocol": "trajuq_pilot_quality_gate_v2",
            "status": "PASS",
            "git_commit": EXPERIMENT_COMMIT,
            "raw": str(raw),
            "raw_sha256": sha256_file(raw),
            "scored": str(scored),
            "scored_sha256": sha256_file(scored),
            "raw_records": 50,
            "scored_records": 50,
            "adapters": ["llada"],
            "model_revisions": ["rev-llada"],
        },
    )

    review_sheet = tmp_path / "pilot_review.md"
    review_sheet.write_text("# reviewed pilot traces\n", encoding="utf-8")
    llada_parity = Path(parity["llada"])
    write_json(
        manual,
        {
            "protocol": "trajuq_pilot_manual_gate_v2",
            "status": "PASS",
            "git_commit": EXPERIMENT_COMMIT,
            "review_sheet": str(review_sheet),
            "review_sheet_sha256": sha256_file(review_sheet),
            "parity_artifact": str(llada_parity),
            "parity_artifact_sha256": sha256_file(llada_parity),
            "pilot_quality": str(quality),
            "pilot_quality_sha256": sha256_file(quality),
            "reviewed_records": 10,
            "checks": {
                "trajectory_semantics": True,
                "scorer_audit": True,
                "parity_artifact_reviewed": True,
                "no_leakage_observed": True,
            },
        },
    )

    write_json(
        math_audit,
        {
            "protocol": "trajuq_math500_scorer_audit_v2",
            "status": "PASS",
            "git_commit": EXPERIMENT_COMMIT,
            "failure_count": 0,
        },
    )
    math_review_sheet = tmp_path / "math_review.md"
    math_review_sheet.write_text("# reviewed math scorer examples\n", encoding="utf-8")
    write_json(
        math_manual,
        {
            "protocol": "trajuq_math500_manual_scorer_gate_v2",
            "status": "PASS",
            "git_commit": EXPERIMENT_COMMIT,
            "automatic_audit": str(math_audit),
            "automatic_audit_sha256": sha256_file(math_audit),
            "automatic_audit_git_commit": EXPERIMENT_COMMIT,
            "review_sheet": str(math_review_sheet),
            "review_sheet_sha256": sha256_file(math_review_sheet),
            "reviewed_records": 25,
            "checks": {
                "answer_extraction": True,
                "symbolic_equivalence": True,
                "no_manual_label_flips": True,
            },
        },
    )

    return {
        "config": config,
        "ledger": ledger,
        "cpu": cpu,
        "quality": quality,
        "manual": manual,
        "math_audit": math_audit,
        "math_manual": math_manual,
        "parity": parity,
    }


def evaluate(case: dict) -> dict:
    return evaluate_readiness(
        str(case["config"]),
        str(case["quality"]),
        str(case["manual"]),
        case["parity"],
        str(case["math_audit"]),
        str(case["math_manual"]),
        str(case["cpu"]),
        str(case["ledger"]),
    )


def test_primary_readiness_passes_only_with_all_bound_gates(tmp_path):
    case = build_case(tmp_path)
    result = evaluate(case)
    assert result["status"] == "PASS"
    assert result["experiment_git_commit"] == EXPERIMENT_COMMIT
    assert not result["failures"]


def test_primary_readiness_rejects_stale_cpu_and_parity_commit(tmp_path):
    case = build_case(tmp_path, ("llada",))
    write_cpu_pass(case["cpu"], commit="old-commit")
    write_parity(
        Path(case["parity"]["llada"]),
        "llada",
        "rev-llada",
        commit="old-commit",
    )

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("frozen experiment commit" in item for item in result["failures"])
    assert any("frozen revision/commit" in item for item in result["failures"])


def test_primary_readiness_rejects_parity_revision_not_in_ledger(tmp_path):
    case = build_case(tmp_path, ("llada",))
    write_parity(
        Path(case["parity"]["llada"]),
        "llada",
        "wrong-revision",
    )

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("frozen revision/commit" in item for item in result["failures"])


def test_primary_readiness_rejects_mutated_pilot_artifact(tmp_path):
    case = build_case(tmp_path, ("llada",))
    quality = json.loads(case["quality"].read_text(encoding="utf-8"))
    Path(quality["raw"]).write_text('{"mutated":true}\n', encoding="utf-8")

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("hash-bound PASS" in item for item in result["failures"])


def test_primary_readiness_rejects_stale_human_review_binding(tmp_path):
    case = build_case(tmp_path, ("llada",))
    Path(case["parity"]["llada"]).write_text(
        Path(case["parity"]["llada"]).read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("human pilot-review gate" in item for item in result["failures"])


def test_primary_readiness_rejects_math_audit_from_other_commit(tmp_path):
    case = build_case(tmp_path, ("llada",))
    audit = json.loads(case["math_audit"].read_text(encoding="utf-8"))
    audit["git_commit"] = "other-commit"
    write_json(case["math_audit"], audit)

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("automatic scorer audit" in item for item in result["failures"])
