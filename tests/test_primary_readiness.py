import json
from pathlib import Path

import yaml

from scripts.primary_readiness import evaluate_readiness


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_primary_readiness_passes_only_with_all_gates(tmp_path):
    config = tmp_path / "primary.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "primary": [
                        "configs/models/llada.yaml",
                        "configs/models/sdar.yaml",
                        "configs/models/dream.yaml",
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    quality = tmp_path / "quality.json"
    manual = tmp_path / "manual.json"
    write_json(quality, {"status": "PASS", "raw_records": 50})
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

    parity = {}
    for model in ("llada", "sdar", "dream"):
        path = tmp_path / f"{model}.json"
        write_json(
            path,
            {
                "status": "PASS",
                "mismatch_count": 0,
                "shared_records": 20,
            },
        )
        parity[model] = str(path)

    math_audit = tmp_path / "math_audit.json"
    math_manual = tmp_path / "math_manual.json"
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
    result = evaluate_readiness(
        str(config),
        str(quality),
        str(manual),
        parity,
        str(math_audit),
        str(math_manual),
    )
    assert result["status"] == "PASS"
    assert not result["failures"]


def test_primary_readiness_fails_when_model_parity_missing(tmp_path):
    config = tmp_path / "primary.yaml"
    config.write_text(
        yaml.safe_dump({"models": {"primary": ["configs/models/llada.yaml"]}}),
        encoding="utf-8",
    )
    quality = tmp_path / "quality.json"
    manual = tmp_path / "manual.json"
    write_json(quality, {"status": "PASS", "raw_records": 50})
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
    math_audit = tmp_path / "math_audit.json"
    math_manual = tmp_path / "math_manual.json"
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
    result = evaluate_readiness(
        str(config),
        str(quality),
        str(manual),
        {},
        str(math_audit),
        str(math_manual),
    )
    assert result["status"] == "FAIL"



def test_primary_readiness_fails_without_math500_manual_gate(tmp_path):
    config = tmp_path / "primary.yaml"
    config.write_text(
        yaml.safe_dump({"models": {"primary": ["configs/models/llada.yaml"]}}),
        encoding="utf-8",
    )
    quality = tmp_path / "quality.json"
    manual = tmp_path / "manual.json"
    parity_path = tmp_path / "llada.json"
    math_audit = tmp_path / "math_audit.json"
    math_manual = tmp_path / "math_manual.json"

    write_json(quality, {"status": "PASS", "raw_records": 50})
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
    write_json(
        parity_path,
        {"status": "PASS", "mismatch_count": 0, "shared_records": 20},
    )
    write_json(math_audit, {"status": "PASS", "failure_count": 0})
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

    result = evaluate_readiness(
        str(config),
        str(quality),
        str(manual),
        {"llada": str(parity_path)},
        str(math_audit),
        str(math_manual),
    )
    assert result["status"] == "FAIL"
    assert any("MATH-500 manual" in item for item in result["failures"])
