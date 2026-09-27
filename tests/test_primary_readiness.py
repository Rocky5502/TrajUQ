import hashlib
import json
from pathlib import Path

import yaml

from scripts.primary_readiness import evaluate_readiness
from trajuq.datasets import NORMALIZER_VERSION


EXPERIMENT_COMMIT = "abc123frozen"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_cpu_pass(path: Path, commit: str = EXPERIMENT_COMMIT) -> None:
    steps = []
    for name in ("compileall", "pytest"):
        log = path.parent / f"{name}.log"
        log.write_text(f"{name} ok\n", encoding="utf-8")
        steps.append(
            {
                "name": name,
                "status": "PASS",
                "log": str(log),
                "log_sha256": sha256_file(log),
            }
        )
    write_json(
        path,
        {
            "protocol": "trajuq_cpu_verification_v1",
            "status": "PASS",
            "git_commit": commit,
            "git_clean": True,
            "expected_steps": ["compileall", "pytest"],
            "completed_steps": ["compileall", "pytest"],
            "steps": steps,
        },
    )


def write_gated_access_pass(
    path: Path,
    ledger_path: Path,
    *,
    commit: str = EXPERIMENT_COMMIT,
) -> None:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    gated = [
        item
        for item in ledger.get("datasets", [])
        if item.get("gated") is True
    ]
    write_json(
        path,
        {
            "protocol": "trajuq_gated_dataset_access_v1",
            "status": "PASS",
            "git_commit": commit,
            "git_clean": True,
            "revision_ledger": str(ledger_path),
            "revision_ledger_sha256": sha256_file(ledger_path),
            "gated_dataset_names": sorted(
                str(item["name"]) for item in gated
            ),
            "datasets": [
                {
                    "name": str(item["name"]),
                    "repo_id": str(item["repo_id"]),
                    "resolved_sha": str(item["resolved_sha"]),
                    "status": "PASS",
                    "record_accessed": True,
                }
                for item in gated
            ],
            "failure_count": 0,
            "failures": [],
        },
    )


def write_dataset_contracts_pass(
    path: Path,
    ledger_path: Path,
    *,
    commit: str = EXPERIMENT_COMMIT,
) -> None:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    datasets = list(ledger.get("datasets", []))
    names = sorted(str(item["name"]) for item in datasets)
    write_json(
        path,
        {
            "protocol": "trajuq_dataset_contracts_v1",
            "status": "PASS",
            "git_commit": commit,
            "git_clean": True,
            "revision_ledger": str(ledger_path),
            "revision_ledger_sha256": sha256_file(ledger_path),
            "normalizer_version": NORMALIZER_VERSION,
            "expected_datasets": names,
            "observed_datasets": names,
            "results": [
                {
                    "dataset": str(item["name"]),
                    "repo_id": str(item["repo_id"]),
                    "resolved_revision": str(item["resolved_sha"]),
                    "normalizer_version": NORMALIZER_VERSION,
                    "scorer_smoke_pass": True,
                    "source_row_content_stored": False,
                    "status": "PASS",
                    "failures": [],
                }
                for item in datasets
            ],
            "failure_count": 0,
            "failures": [],
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
    manifest = path.parent / "evaluation_splits.json"
    manifest.write_text(
        json.dumps({"datasets": {"gsm8k": {"hf_id": "openai/gsm8k"}}}),
        encoding="utf-8",
    )
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
            "dataset_manifest": str(manifest),
            "dataset_manifest_sha256": sha256_file(manifest),
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
    reference = path.with_name(path.stem + ".reference.jsonl")
    instrumented = path.with_name(path.stem + ".instrumented.jsonl")
    reference.write_text('{"example_id":"x"}\n', encoding="utf-8")
    instrumented.write_text('{"example_id":"x"}\n', encoding="utf-8")
    write_json(
        instrumented.with_suffix(instrumented.suffix + ".audit.json"),
        {
            "protocol": "trajuq_run_audit_v1",
            "git_commit": commit,
            "git_clean": True,
            "status": "PASS",
            "stage": "raw",
            "sha256": sha256_file(instrumented),
        },
    )

    write_json(
        path,
        {
            "protocol": "trajuq_exact_parity_v3",
            "status": "PASS",
            "git_commit": commit,
            "git_clean": True,
            "reference": str(reference),
            "reference_sha256": sha256_file(reference),
            "instrumented": str(instrumented),
            "instrumented_sha256": sha256_file(instrumented),
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
    gated_access = tmp_path / "gated_access.json"
    dataset_contracts = tmp_path / "dataset_contracts.json"
    quality = tmp_path / "quality.json"
    manual = tmp_path / "manual.json"
    math_audit = tmp_path / "math_audit.json"
    math_manual = tmp_path / "math_manual.json"

    write_config(config, models)
    revisions = write_revision_ledger(ledger, config, models)
    write_cpu_pass(cpu)
    write_gated_access_pass(gated_access, ledger)
    write_dataset_contracts_pass(dataset_contracts, ledger)

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
        raw.with_suffix(raw.suffix + ".audit.json"),
        {
            "protocol": "trajuq_run_audit_v1",
            "git_commit": EXPERIMENT_COMMIT,
            "git_clean": True,
            "status": "PASS",
            "stage": "raw",
            "sha256": sha256_file(raw),
        },
    )
    write_json(
        scored.with_suffix(scored.suffix + ".audit.json"),
        {
            "protocol": "trajuq_run_audit_v1",
            "git_commit": EXPERIMENT_COMMIT,
            "git_clean": True,
            "status": "PASS",
            "stage": "scored",
            "sha256": sha256_file(scored),
        },
    )
    write_json(
        quality,
        {
            "protocol": "trajuq_pilot_quality_gate_v2",
            "status": "PASS",
            "git_commit": EXPERIMENT_COMMIT,
            "git_clean": True,
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
    review_sheet.write_text(
        "\n".join(
            [
                "# reviewed pilot traces",
                f"- scored artifact: {scored}",
                f"- scored SHA-256: {sha256_file(scored)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    llada_parity = Path(parity["llada"])
    write_json(
        manual,
        {
            "protocol": "trajuq_pilot_manual_gate_v2",
            "status": "PASS",
            "git_commit": EXPERIMENT_COMMIT,
            "git_clean": True,
            "review_sheet": str(review_sheet),
            "review_sheet_sha256": sha256_file(review_sheet),
            "review_scored_artifact": str(scored),
            "review_scored_sha256": sha256_file(scored),
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
            "git_clean": True,
            "failure_count": 0,
            "dependency_versions": {
                "math-verify": "0.9.0",
                "antlr4-python3-runtime": "4.13.2",
            },
        },
    )
    math_scored = tmp_path / "runs" / "llada_math500_scorer_pilot.scored.jsonl"
    math_scored.parent.mkdir(parents=True, exist_ok=True)
    math_scored.write_text('{"example_id":"m"}\n', encoding="utf-8")
    write_json(
        math_scored.with_suffix(math_scored.suffix + ".audit.json"),
        {
            "protocol": "trajuq_run_audit_v1",
            "git_commit": EXPERIMENT_COMMIT,
            "git_clean": True,
            "status": "PASS",
            "stage": "scored",
            "sha256": sha256_file(math_scored),
        },
    )
    math_input = tmp_path / "data" / "processed" / "math500" / "calibration.jsonl"
    math_input.parent.mkdir(parents=True, exist_ok=True)
    math_input.write_text('{"id":"m"}\n', encoding="utf-8")
    math_review_sheet = tmp_path / "math_review.md"
    math_review_sheet.write_text(
        "\n".join(
            [
                "# reviewed math scorer examples",
                f"- scored artifact: {math_scored}",
                f"- scored SHA-256: {sha256_file(math_scored)}",
                f"- normalized input: {math_input}",
                f"- input SHA-256: {sha256_file(math_input)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json(
        math_manual,
        {
            "protocol": "trajuq_math500_manual_scorer_gate_v2",
            "status": "PASS",
            "git_commit": EXPERIMENT_COMMIT,
            "git_clean": True,
            "automatic_audit": str(math_audit),
            "automatic_audit_sha256": sha256_file(math_audit),
            "automatic_audit_git_commit": EXPERIMENT_COMMIT,
            "review_sheet": str(math_review_sheet),
            "review_sheet_sha256": sha256_file(math_review_sheet),
            "review_scored_artifact": str(math_scored),
            "review_scored_sha256": sha256_file(math_scored),
            "review_input": str(math_input),
            "review_input_sha256": sha256_file(math_input),
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
        "gated_access": gated_access,
        "dataset_contracts": dataset_contracts,
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
        str(case["gated_access"]),
        str(case["dataset_contracts"]),
    )


def test_primary_readiness_passes_only_with_all_bound_gates(tmp_path):
    case = build_case(tmp_path)
    result = evaluate(case)
    assert result["status"] == "PASS"
    assert result["experiment_git_commit"] == EXPERIMENT_COMMIT
    assert result["source_bindings"]
    bound_paths = {item["path"] for item in result["source_bindings"]}
    assert str(case["cpu"]) in bound_paths
    assert str(case["ledger"]) in bound_paths
    assert str(case["dataset_contracts"]) in bound_paths
    assert str(case["quality"]) in bound_paths
    assert str(case["manual"]) in bound_paths
    assert str(case["math_audit"]) in bound_paths
    assert str(case["math_manual"]) in bound_paths
    assert not result["failures"]


def test_primary_readiness_rejects_missing_gated_access_gate(tmp_path):
    case = build_case(tmp_path, ("llada",))
    case["gated_access"].unlink()

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any(
        "gated dataset access artifact unavailable" in item
        for item in result["failures"]
    )


def test_primary_readiness_rejects_stale_gated_access_commit(tmp_path):
    case = build_case(tmp_path, ("llada",))
    payload = json.loads(
        case["gated_access"].read_text(encoding="utf-8")
    )
    payload["git_commit"] = "other-commit"
    write_json(case["gated_access"], payload)

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any(
        "gated primary dataset access" in item
        for item in result["failures"]
    )


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
    assert any("hash/audit-bound PASS" in item for item in result["failures"])


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


def test_primary_readiness_rejects_mutated_dataset_manifest(tmp_path):
    case = build_case(tmp_path, ("llada",))
    ledger = json.loads(case["ledger"].read_text(encoding="utf-8"))
    Path(ledger["dataset_manifest"]).write_text(
        json.dumps({"datasets": {"gsm8k": {"hf_id": "changed/repo"}}}),
        encoding="utf-8",
    )

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("dataset manifest hash differs" in item for item in result["failures"])


def test_primary_readiness_can_enforce_current_clean_checkout(tmp_path):
    case = build_case(tmp_path, ("llada",))
    result = evaluate_readiness(
        str(case["config"]),
        str(case["quality"]),
        str(case["manual"]),
        case["parity"],
        str(case["math_audit"]),
        str(case["math_manual"]),
        str(case["cpu"]),
        str(case["ledger"]),
        str(case["gated_access"]),
        enforce_current_git=True,
        current_git_commit="different-commit",
        current_git_clean=False,
    )

    assert result["status"] == "FAIL"
    assert any(
        "current git commit differs from frozen experiment commit" in item
        for item in result["failures"]
    )
    assert any(
        "current git worktree is not clean" in item
        for item in result["failures"]
    )


def test_primary_readiness_rejects_dirty_human_attestations(tmp_path):
    case = build_case(tmp_path, ("llada",))

    manual = json.loads(case["manual"].read_text(encoding="utf-8"))
    manual["git_clean"] = False
    write_json(case["manual"], manual)

    math_manual = json.loads(case["math_manual"].read_text(encoding="utf-8"))
    math_manual["git_clean"] = False
    write_json(case["math_manual"], math_manual)

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("human pilot-review gate" in item for item in result["failures"])
    assert any("MATH-500 manual scorer review" in item for item in result["failures"])


def test_primary_readiness_rejects_math_dependency_version_drift(tmp_path):
    case = build_case(tmp_path, ("llada",))
    audit = json.loads(case["math_audit"].read_text(encoding="utf-8"))
    audit["dependency_versions"]["math-verify"] = "0.8.0"
    write_json(case["math_audit"], audit)

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("automatic scorer audit" in item for item in result["failures"])


def test_primary_readiness_rejects_mutated_parity_source(tmp_path):
    case = build_case(tmp_path, ("llada",))
    parity = json.loads(
        Path(case["parity"]["llada"]).read_text(encoding="utf-8")
    )
    Path(parity["instrumented"]).write_text(
        '{"example_id":"tampered"}\n',
        encoding="utf-8",
    )

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("hash/audit-bound evidence" in item for item in result["failures"])


def test_primary_readiness_rejects_dirty_parity_artifact(tmp_path):
    case = build_case(tmp_path, ("llada",))
    parity_path = Path(case["parity"]["llada"])
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity["git_clean"] = False
    write_json(parity_path, parity)

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("hash/audit-bound evidence" in item for item in result["failures"])


def test_primary_readiness_rejects_dirty_pilot_quality_artifact(tmp_path):
    case = build_case(tmp_path, ("llada",))
    quality = json.loads(case["quality"].read_text(encoding="utf-8"))
    quality["git_clean"] = False
    write_json(case["quality"], quality)

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("pilot quality gate" in item for item in result["failures"])


def test_primary_readiness_rejects_dirty_math_scorer_audit(tmp_path):
    case = build_case(tmp_path, ("llada",))
    audit = json.loads(case["math_audit"].read_text(encoding="utf-8"))
    audit["git_clean"] = False
    write_json(case["math_audit"], audit)

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("automatic scorer audit" in item for item in result["failures"])


def test_primary_readiness_rejects_missing_or_stale_pilot_audits(tmp_path):
    case = build_case(tmp_path, ("llada",))
    quality = json.loads(case["quality"].read_text(encoding="utf-8"))
    raw = Path(quality["raw"])
    raw_audit = raw.with_suffix(raw.suffix + ".audit.json")
    raw_audit.unlink()

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("hash/audit-bound PASS" in item for item in result["failures"])

    write_json(
        raw_audit,
        {
            "protocol": "trajuq_run_audit_v1",
            "git_commit": EXPERIMENT_COMMIT,
            "git_clean": True,
            "status": "PASS",
            "stage": "raw",
            "sha256": "wrong",
        },
    )
    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("hash/audit-bound PASS" in item for item in result["failures"])


def test_primary_readiness_rejects_pilot_review_source_drift(tmp_path):
    case = build_case(tmp_path, ("llada",))
    manual = json.loads(case["manual"].read_text(encoding="utf-8"))
    manual["review_scored_sha256"] = "wrong"
    write_json(case["manual"], manual)

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("human pilot-review gate" in item for item in result["failures"])


def test_primary_readiness_rejects_math_review_source_drift(tmp_path):
    case = build_case(tmp_path, ("llada",))
    manual = json.loads(case["math_manual"].read_text(encoding="utf-8"))
    Path(manual["review_scored_artifact"]).write_text(
        '{"example_id":"tampered"}\n',
        encoding="utf-8",
    )

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("MATH-500 manual scorer review" in item for item in result["failures"])


def test_primary_readiness_rejects_missing_parity_raw_audit(tmp_path):
    case = build_case(tmp_path, ("llada",))
    parity = json.loads(
        Path(case["parity"]["llada"]).read_text(encoding="utf-8")
    )
    instrumented = Path(parity["instrumented"])
    instrumented.with_suffix(instrumented.suffix + ".audit.json").unlink()

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("hash/audit-bound evidence" in item for item in result["failures"])


def test_primary_readiness_rejects_run_audit_from_other_commit(tmp_path):
    case = build_case(tmp_path, ("llada",))
    quality = json.loads(case["quality"].read_text(encoding="utf-8"))
    scored = Path(quality["scored"])
    audit_path = scored.with_suffix(scored.suffix + ".audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["git_commit"] = "other-commit"
    write_json(audit_path, audit)

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any("hash/audit-bound PASS" in item for item in result["failures"])


def test_primary_readiness_rejects_mutated_cpu_log(tmp_path):
    case = build_case(tmp_path, ("llada",))
    cpu = json.loads(case["cpu"].read_text(encoding="utf-8"))
    log_path = Path(cpu["steps"][0]["log"])
    log_path.write_text("tampered\n", encoding="utf-8")

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert result["checks"]["cpu_verification"]["log_hashes_ok"] is False
    assert any("CPU repository verification" in item for item in result["failures"])


def test_primary_readiness_rejects_missing_dataset_contract_gate(tmp_path):
    case = build_case(tmp_path, ("llada",))
    case["dataset_contracts"].unlink()

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any(
        "dataset contract artifact unavailable" in item
        for item in result["failures"]
    )


def test_primary_readiness_rejects_dataset_contract_revision_drift(tmp_path):
    case = build_case(tmp_path, ("llada",))
    payload = json.loads(
        case["dataset_contracts"].read_text(encoding="utf-8")
    )
    payload["results"][0]["resolved_revision"] = "other-revision"
    write_json(case["dataset_contracts"], payload)

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any(
        "normalizer/scorer contracts" in item
        for item in result["failures"]
    )


def test_primary_readiness_rejects_dataset_repo_identity_drift(tmp_path):
    case = build_case(tmp_path, ("llada",))
    ledger = json.loads(case["ledger"].read_text(encoding="utf-8"))
    ledger["datasets"][0]["repo_id"] = "other/gsm8k"
    write_json(case["ledger"], ledger)
    write_gated_access_pass(case["gated_access"], case["ledger"])
    write_dataset_contracts_pass(case["dataset_contracts"], case["ledger"])

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any(
        "revision-ledger repo_id differs from dataset manifest" in item
        for item in result["failures"]
    )


def test_primary_readiness_rejects_frozen_model_identity_drift(tmp_path):
    case = build_case(tmp_path, ("llada",))
    ledger = json.loads(case["ledger"].read_text(encoding="utf-8"))
    frozen_path = Path(ledger["frozen_model_configs"]["llada"])
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    frozen["model_id"] = "other/model"
    frozen_path.write_text(
        yaml.safe_dump(frozen),
        encoding="utf-8",
    )
    ledger["frozen_model_config_sha256"]["llada"] = sha256_file(frozen_path)
    write_json(case["ledger"], ledger)
    write_gated_access_pass(case["gated_access"], case["ledger"])
    write_dataset_contracts_pass(case["dataset_contracts"], case["ledger"])

    result = evaluate(case)
    assert result["status"] == "FAIL"
    assert any(
        "frozen config model_id differs from revision ledger" in item
        for item in result["failures"]
    )
