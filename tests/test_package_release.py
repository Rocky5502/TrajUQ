from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

from scripts.package_release import build_release


def make_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "paper").mkdir()
    (root / "scripts" / "run.py").write_text("print('x')\n", encoding="utf-8")
    (root / "results" / "evidence.json").write_text("{}\n", encoding="utf-8")
    (root / "paper" / "main.tex").write_text("private manuscript\n", encoding="utf-8")
    (root / ".project-id").write_text(
        "TrajUQ\nRocky5502/TrajUQ\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/Rocky5502/TrajUQ.git"],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    return root


def test_release_archive_is_deterministic_and_excludes_paper(tmp_path: Path) -> None:
    root = make_repository(tmp_path)
    first = root / "dist-a"
    second = root / "dist-b"
    payload_a = build_release(root=root, output_dir=first, release_date="20261024")
    payload_b = build_release(root=root, output_dir=second, release_date="20261024")

    assert payload_a["archive"] == payload_b["archive"]
    name = "TrajUQ_CODE_20261024.zip"
    assert (first / name).read_bytes() == (second / name).read_bytes()

    with zipfile.ZipFile(first / name) as archive:
        names = set(archive.namelist())
        internal = json.loads(archive.read("SHA256_MANIFEST.json"))

    assert "scripts/run.py" in names
    assert "results/evidence.json" in names
    assert not any(path.startswith("paper/") for path in names)
    assert internal["kind"] == "CODE"


def test_release_manifest_records_archive_sha256(tmp_path: Path) -> None:
    root = make_repository(tmp_path)
    output = root / "dist"
    payload = build_release(root=root, output_dir=output, release_date="20261024")
    persisted = json.loads(
        (output / "TrajUQ_BUILD_20261024.json").read_text(encoding="utf-8")
    )
    assert persisted == {key: value for key, value in payload.items() if key != "manifest"}
    assert persisted["archive"]["kind"] == "CODE"
