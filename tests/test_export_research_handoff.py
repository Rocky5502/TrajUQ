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
    (root / "manifests").mkdir()
    (root / "docs").mkdir()
    (root / "dist").mkdir()

    required = {
        "README.md": "# TrajUQ\n",
        "EXPERIMENT_FREEZE_CHECKLIST.md": "freeze checklist\n",
        "configs/project_status.yaml": "project: {}\n",
        "configs/experiments/primary.yaml": "x: 1\n",
        "manifests/models.json": "{}\n",
        "manifests/datasets.json": "{}\n",
        "manifests/environment.json": "{}\n",
        "docs/REVIEWER_GUIDE.md": "reviewer guide\n",
        "docs/RESEARCH_QUESTIONS.md": "rqs\n",
        "docs/EXPERIMENT_PROTOCOL.md": "protocol\n",
        "docs/REPRODUCIBILITY.md": "repro\n",
        "docs/A100_EXECUTION.md": "a100\n",
        "docs/EVIDENCE_MAP.md": "evidence\n",
    }
    for rel, content in required.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    (root / "results" / "metrics.json").write_text(
        json.dumps({"auroc": 0.5}), encoding="utf-8"
    )
    (root / "results" / "snapshot.md").write_text("metadata\n", encoding="utf-8")
    (root / "results" / "predictions.jsonl").write_text(
        '{"prompt":"secret-ish"}\n', encoding="utf-8"
    )
    (root / "runs" / "raw.json").write_text('{"raw": true}\n', encoding="utf-8")
    (root / "data" / "processed" / "dataset.json").write_text(
        '{"answer": 42}\n', encoding="utf-8"
    )
    return root


def fake_git_value(root: Path, *args: str) -> str:
    del root
    return "abc123" if args[0] == "rev-parse" else ""


def test_bundle_excludes_raw_jsonl_runs_processed_data_and_manuscript(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    monkeypatch.setattr(handoff, "git_value", fake_git_value)
    output = root / "dist" / "handoff.zip"

    payload = handoff.build_handoff(root, output)

    assert payload["status"] == "PASS"
    assert payload["protocol"] == "trajuq_research_handoff_v2"
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("MANIFEST.json"))

    assert "results/metrics.json" in names
    assert "results/snapshot.md" in names
    assert "results/predictions.jsonl" not in names
    assert "runs/raw.json" not in names
    assert "data/processed/dataset.json" not in names
    assert "docs/REVIEWER_GUIDE.md" in names
    assert "docs/EVIDENCE_MAP.md" in names
    assert not any(name.startswith("paper/") for name in names)
    assert manifest["privacy"]["manuscript_source_included"] is False


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
    (root / "docs" / "REVIEWER_GUIDE.md").unlink()

    with pytest.raises(ValueError, match="required handoff context missing"):
        handoff.build_handoff(root, root / "dist" / "handoff.zip")


def test_bundle_ignores_pytest_scratch_secret_fixture(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    monkeypatch.setattr(handoff, "git_value", fake_git_value)
    scratch = root / "results" / "pytest_tmp" / "test_secret_fixture"
    scratch.mkdir(parents=True)
    (scratch / "bad.json").write_text(
        json.dumps({"token": "hf_" + "abcdefghijklmnopqrstuvwxyz123456"}),
        encoding="utf-8",
    )
    output = root / "dist" / "handoff.zip"

    payload = handoff.build_handoff(root, output)

    assert payload["status"] == "PASS"
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
    assert not any(name.startswith("results/pytest_") for name in names)


def test_bundle_rejects_secret_like_token(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    monkeypatch.setattr(handoff, "git_value", fake_git_value)
    (root / "results" / "bad.json").write_text(
        json.dumps({"token": "hf_" + "abcdefghijklmnopqrstuvwxyz123456"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="possible Hugging Face token"):
        handoff.build_handoff(root, root / "dist" / "handoff.zip")
