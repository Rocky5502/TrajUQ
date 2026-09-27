from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from scripts.build_overleaf_package import ACL_STYLE_COMMIT, build_package


def make_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    paper = root / "paper"
    figures = paper / "figures" / "generated"
    style = tmp_path / "style"
    figures.mkdir(parents=True)
    style.mkdir()

    (paper / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}x\\end{document}\n",
        encoding="utf-8",
    )
    (paper / "appendix.tex").write_text("appendix\n", encoding="utf-8")
    (paper / "references.bib").write_text(
        "@article{x,title={x},author={x},year={2026}}\n",
        encoding="utf-8",
    )
    (paper / "generated_results.tex").write_text(
        "\\newcommand{\\X}{\\textsc{TBD}}\n",
        encoding="utf-8",
    )
    (figures / "result.pdf").write_bytes(b"%PDF-1.4\nsynthetic\n")
    (style / "acl.sty").write_text(
        "% https://github.com/acl-org/acl-style-files/\n",
        encoding="utf-8",
    )
    (style / "acl_natbib.bst").write_text(
        "% pinned bst\n",
        encoding="utf-8",
    )
    (root / "dist").mkdir()
    return root, style


def test_overleaf_package_is_deterministic_and_complete(tmp_path):
    root, style = make_root(tmp_path)
    first = root / "dist" / "first.zip"
    second = root / "dist" / "second.zip"

    first_payload = build_package(
        root=root,
        output=first,
        style_dir=style,
        git_commit="a" * 40,
    )
    second_payload = build_package(
        root=root,
        output=second,
        style_dir=style,
        git_commit="a" * 40,
    )

    assert first_payload["status"] == "PASS"
    assert first_payload["acl_style_commit"] == ACL_STYLE_COMMIT
    assert first.read_bytes() == second.read_bytes()
    assert ("a" * 40).encode("utf-8") not in first.read_bytes()

    with zipfile.ZipFile(first) as zf:
        names = set(zf.namelist())
        assert {
            "main.tex",
            "appendix.tex",
            "references.bib",
            "generated_results.tex",
            "acl.sty",
            "acl_natbib.bst",
            "PAPER_STATUS.md",
            "MANIFEST.json",
            "figures/generated/result.pdf",
        } <= names


def test_overleaf_package_rejects_identity_leak(tmp_path):
    root, style = make_root(tmp_path)
    identity_url = "https://github.com/" + "Rocky" + "5502/TrajUQ\n"
    (root / "paper" / "appendix.tex").write_text(
        identity_url,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="development repository"):
        build_package(
            root=root,
            output=root / "dist" / "bad.zip",
            style_dir=style,
            git_commit="b" * 40,
        )


def test_overleaf_package_rejects_non_pdf_generated_figure(tmp_path):
    root, style = make_root(tmp_path)
    (root / "paper" / "figures" / "generated" / "bad.pdf").write_bytes(
        b"not a pdf"
    )

    with pytest.raises(ValueError, match="invalid generated PDF"):
        build_package(
            root=root,
            output=root / "dist" / "bad.zip",
            style_dir=style,
            git_commit="c" * 40,
        )
