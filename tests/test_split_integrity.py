import hashlib
import json
from pathlib import Path

from trajuq.data_integrity import audit_split_pair
from trajuq.datasets import NORMALIZER_VERSION


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
