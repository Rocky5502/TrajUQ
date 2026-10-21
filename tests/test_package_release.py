from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

from scripts.package_release import build_release


def make_repository(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    paper = root / "paper"
    figures = paper / "figures"
    styles = tmp_path / "styles"
    figures.mkdir(parents=True)
    styles.mkdir()
    (root / "scripts").mkdir()
    (root / "results").mkdir()
    (paper / "main.tex").write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\\input{section}"
        "\\includegraphics{figures/chart.pdf}"
        "\\bibliographystyle{tmlr}\\bibliography{references}\\end{document}\n",
        encoding="utf-8",
    )
    (paper / "section.tex").write_text("text\n", encoding="utf-8")
    (paper / "references.bib").write_text("@misc{x,title={X}}\n", encoding="utf-8")
    (figures / "chart.pdf").write_bytes(b"%PDF-1.4\n")
    (paper / "private-analysis.json").write_text("{}\n", encoding="utf-8")
    (root / "scripts" / "run.py").write_text("print('x')\n", encoding="utf-8")
    (root / "results" / "evidence.json").write_text("{}\n", encoding="utf-8")
    for name in ("tmlr.sty", "tmlr.bst", "fancyhdr.sty"):
        (styles / name).write_text(f"% {name}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
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
    return root, styles


def test_release_archives_are_deterministic_and_disjoint(tmp_path: Path) -> None:
    root, styles = make_repository(tmp_path)
    first = root / "dist-a"
    second = root / "dist-b"
    payload_a = build_release(
        root=root, output_dir=first, release_date="20261020", style_dir=styles
    )
    payload_b = build_release(
        root=root, output_dir=second, release_date="20261020", style_dir=styles
    )

    assert payload_a["archives"] == payload_b["archives"]
    for name in (
        "TrajUQ_TMLR_CODE_20261020.zip",
        "TrajUQ_TMLR_OVERLEAF_20261020.zip",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    with zipfile.ZipFile(first / "TrajUQ_TMLR_CODE_20261020.zip") as archive:
        code_names = set(archive.namelist())
    with zipfile.ZipFile(first / "TrajUQ_TMLR_OVERLEAF_20261020.zip") as archive:
        paper_names = set(archive.namelist())
        internal = json.loads(archive.read("SHA256_MANIFEST.json"))

    assert "scripts/run.py" in code_names
    assert "results/evidence.json" in code_names
    assert not any(name.startswith("paper/") for name in code_names)
    assert "main.tex" in paper_names
    assert "figures/chart.pdf" in paper_names
    assert "private-analysis.json" not in paper_names
    assert not any(name.startswith("results/") for name in paper_names)
    assert internal["kind"] == "OVERLEAF"


def test_release_manifest_records_archive_sha256(tmp_path: Path) -> None:
    root, styles = make_repository(tmp_path)
    output = root / "dist"
    payload = build_release(
        root=root, output_dir=output, release_date="20261020", style_dir=styles
    )
    persisted = json.loads(
        (output / "TrajUQ_TMLR_BUILD_20261020.json").read_text(encoding="utf-8")
    )
    assert persisted == {key: value for key, value in payload.items() if key != "manifest"}
    assert {item["kind"] for item in persisted["archives"]} == {"CODE", "OVERLEAF"}
