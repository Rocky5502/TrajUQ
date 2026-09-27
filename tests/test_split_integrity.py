import hashlib
import json
from pathlib import Path

from trajuq.data_integrity import audit_split_pair
from trajuq.datasets import NORMALIZER_VERSION, SPLIT_SELECTION_VERSION


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_split_audit_detects_exact_prompt_overlap(tmp_path):
    cal = tmp_path / "calibration.jsonl"
    test = tmp_path / "test.jsonl"
    manifest = tmp_path / "split_manifest.json"
    write_jsonl(
        cal,
        [{"id": "d:train:1", "dataset": "d", "prompt": "same", "reference": "A"}],
    )
    write_jsonl(
        test,
        [{"id": "d:test:1", "dataset": "d", "prompt": "same", "reference": "A"}],
    )
    manifest.write_text(
        json.dumps(
            {
                "dataset": "d",
                "source_git_commit": "frozen-commit",
                "source_git_clean": True,
                "requested_revision": "sha",
                "normalizer_version": NORMALIZER_VERSION,
                "n_calibration": 1,
                "n_test": 1,
                "calibration_sha256": sha(cal),
                "test_sha256": sha(test),
            }
        ),
        encoding="utf-8",
    )

    result = audit_split_pair(cal, test, manifest)
    assert result["status"] == "FAIL"
    assert result["exact_prompt_overlap_count"] == 1


def test_split_audit_rejects_dirty_or_unbound_preparation(tmp_path):
    cal = tmp_path / "calibration.jsonl"
    test = tmp_path / "test.jsonl"
    manifest = tmp_path / "split_manifest.json"
    write_jsonl(
        cal,
        [{"id": "d:train:1", "dataset": "d", "prompt": "p1", "reference": "A"}],
    )
    write_jsonl(
        test,
        [{"id": "d:test:1", "dataset": "d", "prompt": "p2", "reference": "B"}],
    )
    manifest.write_text(
        json.dumps(
            {
                "dataset": "d",
                "source_git_commit": "",
                "source_git_clean": False,
                "requested_revision": "sha",
                "normalizer_version": NORMALIZER_VERSION,
                "n_calibration": 1,
                "n_test": 1,
                "calibration_sha256": sha(cal),
                "test_sha256": sha(test),
            }
        ),
        encoding="utf-8",
    )

    result = audit_split_pair(cal, test, manifest)
    assert result["status"] == "FAIL"
    assert any("source git commit is missing" in x for x in result["failures"])
    assert any("clean git checkout" in x for x in result["failures"])



def valid_manifest(cal: Path, test: Path) -> dict:
    return {
        "dataset": "d",
        "source_git_commit": "frozen-commit",
        "source_git_clean": True,
        "requested_revision": "sha",
        "normalizer_version": NORMALIZER_VERSION,
        "selection_version": SPLIT_SELECTION_VERSION,
        "pre_generation_cap": {
            "method": SPLIT_SELECTION_VERSION,
            "seed": "2027",
            "calibration_max": 1,
            "test_max": 1,
        },
        "source_counts": {
            "source_total": 5,
            "calibration_before_cap": 2,
            "test_before_cap": 3,
        },
        "n_calibration": 1,
        "n_test": 1,
        "calibration_sha256": sha(cal),
        "test_sha256": sha(test),
    }


def test_split_audit_accepts_consistent_pre_cap_provenance(tmp_path):
    cal = tmp_path / "calibration.jsonl"
    test = tmp_path / "test.jsonl"
    manifest = tmp_path / "split_manifest.json"
    write_jsonl(
        cal,
        [{"id": "d:train:1", "dataset": "d", "prompt": "p1", "reference": "A"}],
    )
    write_jsonl(
        test,
        [{"id": "d:test:1", "dataset": "d", "prompt": "p2", "reference": "B"}],
    )
    manifest.write_text(
        json.dumps(valid_manifest(cal, test)),
        encoding="utf-8",
    )

    result = audit_split_pair(cal, test, manifest)

    assert result["status"] == "PASS"
    assert result["source_counts"]["source_total"] == 5


def test_split_audit_rejects_inconsistent_source_total(tmp_path):
    cal = tmp_path / "calibration.jsonl"
    test = tmp_path / "test.jsonl"
    manifest = tmp_path / "split_manifest.json"
    write_jsonl(
        cal,
        [{"id": "d:train:1", "dataset": "d", "prompt": "p1", "reference": "A"}],
    )
    write_jsonl(
        test,
        [{"id": "d:test:1", "dataset": "d", "prompt": "p2", "reference": "B"}],
    )
    payload = valid_manifest(cal, test)
    payload["source_counts"]["source_total"] = 4
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_split_pair(cal, test, manifest)

    assert result["status"] == "FAIL"
    assert any(
        "source_total differs" in item
        for item in result["failures"]
    )


def test_split_audit_rejects_selected_count_not_implied_by_cap(tmp_path):
    cal = tmp_path / "calibration.jsonl"
    test = tmp_path / "test.jsonl"
    manifest = tmp_path / "split_manifest.json"
    write_jsonl(
        cal,
        [
            {"id": "d:train:1", "dataset": "d", "prompt": "p1", "reference": "A"},
            {"id": "d:train:2", "dataset": "d", "prompt": "p2", "reference": "B"},
        ],
    )
    write_jsonl(
        test,
        [{"id": "d:test:1", "dataset": "d", "prompt": "p3", "reference": "B"}],
    )
    payload = valid_manifest(cal, test)
    payload["pre_generation_cap"]["calibration_max"] = 2
    payload["n_calibration"] = 2
    payload["calibration_sha256"] = sha(cal)
    # Frozen provenance claims three pre-cap calibration rows, so cap=2 implies
    # exactly two output rows; alter to cap=1 to prove the audit fails closed.
    payload["source_counts"] = {
        "source_total": 6,
        "calibration_before_cap": 3,
        "test_before_cap": 3,
    }
    payload["pre_generation_cap"]["calibration_max"] = 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_split_pair(cal, test, manifest)

    assert result["status"] == "FAIL"
    assert any(
        "prepared count 2 differs from frozen cap expectation 1" in item
        for item in result["failures"]
    )
