import hashlib
import json
from pathlib import Path

import yaml

from scripts.validate_frozen_preprimary import validate_frozen_preprimary
from trajuq.datasets import NORMALIZER_VERSION, SPLIT_SELECTION_VERSION


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_frozen_case(tmp_path: Path) -> dict[str, Path]:
    commit = "frozen-commit"
    primary = tmp_path / "primary.yaml"
    manifest = tmp_path / "evaluation_splits.json"
    frozen = tmp_path / "llada.yaml"
    cpu = tmp_path / "cpu.json"
    ledger = tmp_path / "ledger.json"
    gated = tmp_path / "gated.json"
    contracts = tmp_path / "contracts.json"
    log = tmp_path / "pytest.log"

    primary.write_text(
        yaml.safe_dump(
            {
                "models": {"primary": ["configs/models/llada.yaml"]},
                "datasets": ["gsm8k"],
            }
        ),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "datasets": {
                    "gsm8k": {
                        "hf_id": "openai/gsm8k",
                        "strategy": "official",
                        "calibration_split": "train",
                        "test_split": "test",
                        "pre_generation_cap": {
                            "method": SPLIT_SELECTION_VERSION,
                            "seed": "2027",
                            "calibration_max": 500,
                            "test_max": 500,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    frozen.write_text(
        yaml.safe_dump(
            {
                "adapter": "llada",
                "model_id": "example/llada",
                "revision": "model-sha",
                "device": "cuda",
                "dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )
    log.write_text("pytest ok\n", encoding="utf-8")
    write_json(
        cpu,
        {
            "protocol": "trajuq_cpu_verification_v1",
            "status": "PASS",
            "git_commit": commit,
            "git_clean": True,
            "expected_steps": ["pytest"],
            "completed_steps": ["pytest"],
            "steps": [
                {
                    "name": "pytest",
                    "status": "PASS",
                    "log": str(log),
                    "log_sha256": sha(log),
                }
            ],
        },
    )
    write_json(
        ledger,
        {
            "protocol": "trajuq_revision_ledger_v2",
            "status": "PASS",
            "git_commit": commit,
            "primary_config": str(primary),
            "primary_config_sha256": sha(primary),
            "dataset_manifest": str(manifest),
            "dataset_manifest_sha256": sha(manifest),
            "models": [
                {
                    "adapter": "llada",
                    "model_id": "example/llada",
                    "resolved_sha": "model-sha",
                }
            ],
            "datasets": [
                {
                    "name": "gsm8k",
                    "repo_id": "openai/gsm8k",
                    "resolved_sha": "dataset-sha",
                    "gated": False,
                }
            ],
            "frozen_model_configs": {"llada": str(frozen)},
            "frozen_model_config_sha256": {"llada": sha(frozen)},
            "failures": [],
        },
    )
    ledger_sha = sha(ledger)
    write_json(
        gated,
        {
            "protocol": "trajuq_gated_dataset_access_v1",
            "status": "PASS",
            "git_commit": commit,
            "git_clean": True,
            "revision_ledger_sha256": ledger_sha,
            "gated_dataset_names": [],
            "datasets": [],
            "failure_count": 0,
            "failures": [],
        },
    )
    write_json(
        contracts,
        {
            "protocol": "trajuq_dataset_contracts_v1",
            "status": "PASS",
            "git_commit": commit,
            "git_clean": True,
            "revision_ledger_sha256": ledger_sha,
            "normalizer_version": NORMALIZER_VERSION,
            "selection_version": SPLIT_SELECTION_VERSION,
            "contract_sample_count_per_dataset": 4,
            "expected_datasets": ["gsm8k"],
            "observed_datasets": ["gsm8k"],
            "results": [
                {
                    "dataset": "gsm8k",
                    "repo_id": "openai/gsm8k",
                    "resolved_revision": "dataset-sha",
                    "normalizer_version": NORMALIZER_VERSION,
                    "contract_sample_count": 4,
                    "observed_sample_count": 4,
                    "scorer_smoke_pass_count": 4,
                    "scorer_smoke_pass": True,
                    "source_row_content_stored": False,
                    "status": "PASS",
                }
            ],
            "failure_count": 0,
            "failures": [],
        },
    )
    return {
        "cpu": cpu,
        "ledger": ledger,
        "gated": gated,
        "contracts": contracts,
        "log": log,
        "frozen": frozen,
    }


def validate(case: dict[str, Path], *, clean: bool = True):
    return validate_frozen_preprimary(
        cpu_path=case["cpu"],
        ledger_path=case["ledger"],
        gated_path=case["gated"],
        contracts_path=case["contracts"],
        require_current_checkout=True,
        current_head="frozen-commit",
        current_clean=clean,
    )


def test_shared_frozen_preprimary_validator_accepts_bound_case(tmp_path):
    case = build_frozen_case(tmp_path)
    result = validate(case)

    assert result["status"] == "PASS"
    assert result["checks"]["cpu_logs"] is True
    assert result["checks"]["dataset_contracts"] is True
    assert result["checks"]["frozen_model_configs"]["llada"] is True


def test_shared_validator_rejects_cpu_log_drift(tmp_path):
    case = build_frozen_case(tmp_path)
    case["log"].write_text("tampered\n", encoding="utf-8")

    result = validate(case)
    assert result["status"] == "FAIL"
    assert any("CPU verification/log evidence" in x for x in result["failures"])


def test_shared_validator_rejects_frozen_model_config_drift(tmp_path):
    case = build_frozen_case(tmp_path)
    case["frozen"].write_text("adapter: llada\nrevision: changed\n", encoding="utf-8")

    result = validate(case)
    assert result["status"] == "FAIL"
    assert any("frozen model config" in x for x in result["failures"])


def test_shared_validator_rejects_dataset_contract_revision_drift(tmp_path):
    case = build_frozen_case(tmp_path)
    payload = json.loads(case["contracts"].read_text(encoding="utf-8"))
    payload["results"][0]["resolved_revision"] = "other-sha"
    write_json(case["contracts"], payload)

    result = validate(case)
    assert result["status"] == "FAIL"
    assert any("dataset normalizer/scorer contracts" in x for x in result["failures"])


def test_shared_validator_rejects_dirty_checkout(tmp_path):
    case = build_frozen_case(tmp_path)
    result = validate(case, clean=False)

    assert result["status"] == "FAIL"
    assert "current git worktree is not clean" in result["failures"]


def test_shared_validator_rejects_legacy_one_row_dataset_contract(tmp_path):
    case = build_frozen_case(tmp_path)
    payload = json.loads(case["contracts"].read_text(encoding="utf-8"))
    payload["contract_sample_count_per_dataset"] = 1
    payload["results"][0]["contract_sample_count"] = 1
    payload["results"][0]["observed_sample_count"] = 1
    payload["results"][0]["scorer_smoke_pass_count"] = 1
    write_json(case["contracts"], payload)

    result = validate(case)
    assert result["status"] == "FAIL"
    assert any(
        "dataset normalizer/scorer contracts" in x
        for x in result["failures"]
    )



def test_frozen_preprimary_rejects_runtime_dtype_drift(tmp_path):
    case = build_frozen_case(tmp_path)
    frozen = case["frozen"]
    payload = yaml.safe_load(frozen.read_text(encoding="utf-8"))
    payload["dtype"] = "float16"
    frozen.write_text(yaml.safe_dump(payload), encoding="utf-8")

    ledger = json.loads(case["ledger"].read_text(encoding="utf-8"))
    ledger["frozen_model_config_sha256"]["llada"] = sha(frozen)
    case["ledger"].write_text(json.dumps(ledger), encoding="utf-8")
    ledger_sha = sha(case["ledger"])

    for key in ("gated", "contracts"):
        artifact = json.loads(case[key].read_text(encoding="utf-8"))
        artifact["revision_ledger_sha256"] = ledger_sha
        case[key].write_text(json.dumps(artifact), encoding="utf-8")

    result = validate(case)

    assert result["status"] == "FAIL"
    assert result["checks"]["frozen_model_configs"]["llada"] is False
    assert any(
        "frozen model config does not match revision ledger" in item
        for item in result["failures"]
    )
