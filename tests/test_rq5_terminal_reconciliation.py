from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import uuid

import yaml


SCRIPT = Path("scripts/reconcile_rq5_terminal_registry.py")


def load_module():
    spec = importlib.util.spec_from_file_location("rq5_reconcile", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_fixture(tmp_path: Path):
    module = load_module()
    root = tmp_path
    data = root / "data/processed/demo/calibration.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text('{"id":"a"}\n{"id":"b"}\n', encoding="utf-8")
    freeze = {
        "status": "PASS",
        "models": ["provider/model"],
        "decoding": {"repeated_samples": 2},
        "prompt_ledger_rows": 2,
        "partitions": [
            {
                "dataset": "demo",
                "split": "calibration",
                "rows": 2,
                "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            }
        ],
    }
    freeze_path = root / "results/rq5/TMLR_RQ5_BLACKBOX_FREEZE.json"
    write_json(freeze_path, freeze)
    freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    Path(str(freeze_path) + ".sha256").write_text(freeze_sha, encoding="utf-8")
    schema = {"required": ["freeze_sha256", "request_uuid"], "forbidden": ["secret"]}
    write_json(root / "configs/schemas/rq5_api_provenance_v1.json", schema)
    identities = []
    for example_id in ("a", "b"):
        for repeat_index in range(2):
            request_uuid = str(
                uuid.uuid5(
                    module.NAMESPACE,
                    f"{freeze_sha}|provider/model|demo|calibration|{example_id}|{repeat_index}",
                )
            )
            identities.append((request_uuid, example_id, repeat_index))
    valid_id, malformed_id, provider_id, ambiguous_id = identities

    def common(rid, eid, rep):
        return {
            "freeze_sha256": freeze_sha,
            "request_uuid": rid,
            "requested_model": "provider/model",
            "returned_model": "provider/model",
            "dataset": "demo",
            "split": "calibration",
            "example_id": eid,
            "repeat_index": rep,
            "http_status": 200,
        }

    records = root / "results/rq5/api/records/provider__model/demo/calibration"
    write_json(
        records / f"{valid_id[0]}.json",
        {**common(*valid_id), "parsed_final_answer": "x", "verbalized_confidence": 50},
    )
    write_json(
        records / f"{malformed_id[0]}.json",
        {**common(*malformed_id), "parsed_final_answer": None, "verbalized_confidence": None},
    )
    error = {
        "schema": "trajuq_rq5_request_error_v1",
        "freeze_sha256": freeze_sha,
        "request_uuid": provider_id[0],
        "requested_model": "provider/model",
        "dataset": "demo",
        "split": "calibration",
        "example_id": provider_id[1],
        "repeat_index": provider_id[2],
        "confirmed_provider_error": True,
        "will_resubmit": False,
    }
    write_json(
        root / f"results/rq5/api/errors/provider__model/demo/calibration/{provider_id[0]}.json",
        error,
    )
    lock = records / f"{ambiguous_id[0]}.pending"
    write_json(lock, {"request_uuid": ambiguous_id[0]})
    ambiguous = {"freeze_sha256": freeze_sha, "requests": []}
    ambiguous_path = root / "configs/operations/rq5_ambiguous_uuid_registry.yaml"
    ambiguous_path.parent.mkdir(parents=True, exist_ok=True)
    ambiguous_path.write_text(yaml.safe_dump(ambiguous), encoding="utf-8")
    return module, root, freeze_path, ambiguous_path


def test_exact_mutually_exclusive_terminal_reconciliation(tmp_path):
    module, root, freeze_path, ambiguous_path = make_fixture(tmp_path)
    manifest, ledger = module.reconcile(
        root=root,
        freeze_path=freeze_path,
        schema_path=root / "configs/schemas/rq5_api_provenance_v1.json",
        api_root=root / "results/rq5/api",
        ambiguous_registry_path=ambiguous_path,
    )
    assert manifest["status"] == "PASS"
    assert manifest["planned_requests"] == manifest["terminal_requests"] == 4
    assert manifest["counts"] == {
        "valid": 1,
        "malformed": 1,
        "provider_failure": 1,
        "ambiguous_locked": 1,
        "other_terminal": 0,
        "missing_unreconciled": 0,
    }
    assert sum(manifest["counts"].values()) == 4
    assert sum(row["locked_no_replay"] for row in ledger) == 1
    assert manifest["analysis_authorized"] is False
    cell = manifest["cell_missingness"][0]
    assert cell["planned"] == 4
    assert cell["successful"] == 2
    assert cell["schema_valid"] == cell["included"] == 1
    assert cell["excluded"] == 3
    assert cell["excluded_by_reason"] == {
        "malformed": 1,
        "provider_failure": 1,
        "locked_ambiguous": 1,
        "other_terminal": 0,
        "missing_unreconciled": 0,
    }
    assert cell["planned"] == cell["included"] + cell["excluded"]


def test_materialization_refuses_overwrite(tmp_path):
    module = load_module()
    target = tmp_path / "ledger.jsonl"
    module.write_new(target, b"first")
    try:
        module.write_new(target, b"second")
    except FileExistsError:
        pass
    else:
        raise AssertionError("reconciler must refuse artifact overwrite")


def test_check_only_contract_is_network_free_and_non_writing():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--check-only" in text
    assert "urlopen" not in text
    assert "requests." not in text
    assert "preserve_as_missing_never_resubmit" in text
    assert 'with path.open("xb")' in text
