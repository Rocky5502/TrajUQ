import json
import zipfile
from pathlib import Path

import pytest

import scripts.export_research_handoff as handoff


def make_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "data" / "processed").mkdir(parents=True)
    (root / "configs" / "experiments").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "paper").mkdir()
    (root / "dist").mkdir()
    (root / "PROJECT_STATUS.md").write_text("status\n", encoding="utf-8")
    (root / "NEW_CHAT_MASTER_PROMPT.md").write_text("prompt\n", encoding="utf-8")
    (root / "configs" / "project_status.yaml").write_text(
        "project: {}\n",
        encoding="utf-8",
    )
    (root / "configs" / "experiments" / "primary.yaml").write_text(
        "x: 1\n",
        encoding="utf-8",
    )
    (root / "results" / "metrics.json").write_text(
        json.dumps({"auroc": 0.5}),
        encoding="utf-8",
    )
    (root / "results" / "snapshot.md").write_text("metadata\n", encoding="utf-8")
    (root / "results" / "predictions.jsonl").write_text(
        '{"prompt":"secret-ish"}\n',
        encoding="utf-8",
    )
    (root / "runs" / "raw.json").write_text('{"raw": true}\n', encoding="utf-8")
    (root / "data" / "processed" / "dataset.json").write_text(
        '{"answer": 42}\n',
        encoding="utf-8",
    )
    return root


def fake_git_value(root: Path, *args: str) -> str:
    del root
    return "abc123" if args[0] == "rev-parse" else ""


def test_bundle_excludes_raw_jsonl_runs_and_processed_data(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    monkeypatch.setattr(handoff, "git_value", fake_git_value)
    output = root / "dist" / "handoff.zip"

    payload = handoff.build_handoff(root, output)

    assert payload["status"] == "PASS"
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
    assert "results/metrics.json" in names
    assert "results/snapshot.md" in names
    assert "results/predictions.jsonl" not in names
    assert "runs/raw.json" not in names
    assert "data/processed/dataset.json" not in names
    assert "NEW_CHAT_MASTER_PROMPT.md" in names
    assert "MANIFEST.json" in names


def test_bundle_is_deterministic_for_same_state(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    monkeypatch.setattr(handoff, "git_value", fake_git_value)
    first = root / "dist" / "first.zip"
    second = root / "dist" / "second.zip"

    first_payload = handoff.build_handoff(root, first)
    second_payload = handoff.build_handoff(root, second)

    assert first_payload["archive_sha256"] == second_payload["archive_sha256"]
    assert first.read_bytes() == second.read_bytes()


def test_bundle_rejects_missing_required_context(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    monkeypatch.setattr(handoff, "git_value", fake_git_value)
    (root / "NEW_CHAT_MASTER_PROMPT.md").unlink()

    with pytest.raises(ValueError, match="required handoff context missing"):
        handoff.build_handoff(root, root / "dist" / "handoff.zip")


def test_bundle_rejects_secret_like_token(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    monkeypatch.setattr(handoff, "git_value", fake_git_value)
    (root / "results" / "bad.json").write_text(
        json.dumps({"token": "hf_abcdefghijklmnopqrstuvwxyz123456"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="possible Hugging Face token"):
        handoff.build_handoff(root, root / "dist" / "handoff.zip")
