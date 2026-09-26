from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path


def test_anonymous_supplement_builds_and_excludes_identity_paths(tmp_path):
    out = tmp_path / "anon.zip"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_anonymous_supplement.py",
            "--output",
            (
                str(out.relative_to(Path.cwd()))
                if out.is_relative_to(Path.cwd())
                else "dist/test_anon.zip"
            ),
        ],
        check=True,
    )
    archive = out if out.exists() else Path("dist/test_anon.zip")
    assert archive.exists()

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        assert "README.md" in names
        assert "MANIFEST.json" in names
        assert "CITATION.cff" not in names
        assert "docs/A100_EXECUTION.md" not in names
        assert "docs/RESULT_ARTIFACT_MAP.md" in names
        assert "scripts/aggregate_feature_ablation.py" in names
        assert "paper/main.tex" not in names
        assert all(not name.startswith("runs/") for name in names)
        assert all(not name.startswith("results/") for name in names)

        combined = b"\n".join(
            zf.read(name)
            for name in sorted(names)
            if name.endswith((".md", ".py", ".yaml", ".yml", ".toml", ".json", ".ps1"))
        )
        development_owner = b"Rocky" + b"5502"
        assert development_owner not in combined

    if archive == Path("dist/test_anon.zip"):
        archive.unlink()
