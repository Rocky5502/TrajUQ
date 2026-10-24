from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_release_manifest import sha256


ROOT = Path(__file__).resolve().parents[1]


def test_authorized_release_artifact_hashes_match() -> None:
    payload = json.loads((ROOT / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    assert payload["experiment_commit"] == "28002055c828c6910840de6f2105aa6367514144"
    assert payload["pending_science"]
    for item in payload["artifacts"]:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert sha256(path) == item["sha256"], item["path"]
        assert item["status"].startswith("AUTHORIZED")

