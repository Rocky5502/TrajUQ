from __future__ import annotations

import hashlib
import json
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
    build_package(
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


def _write_submission_manifests(root: Path) -> dict[str, Path]:
    paper = root / "paper"
    figures = paper / "figures" / "generated"
    generated = paper / "generated_results.tex"
    generated.write_text(
        "\\newcommand{\\PrimaryTrajAUROC}{0.700}\n",
        encoding="utf-8",
    )
    (paper / "generated_results_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "trajuq_paper_results_render_v2",
                "output_sha256": hashlib.sha256(
                    generated.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    outputs = {}
    paths = {}
    for key, filename in (
        ("primary_auroc_matrix", "primary_auroc_improvement_matrix.pdf"),
        ("family_macro_auroc", "family_macro_auroc_improvement.pdf"),
        ("adaptive_tradeoff", "adaptive_accuracy_cost_tradeoff.pdf"),
    ):
        path = figures / filename
        path.write_bytes(f"%PDF-1.4\\n{key}\\n".encode())
        paths[key] = path
        outputs[key] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    (figures / "result_figure_manifest.json").write_text(
        json.dumps(
            {
                "protocol": "trajuq_paper_result_figures_v2",
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )
    return paths


def test_submission_overleaf_packages_only_manifest_bound_figures(tmp_path):
    root, style = make_root(tmp_path)
    paths = _write_submission_manifests(root)
    stale = root / "paper" / "figures" / "generated" / "stale.pdf"
    stale.write_bytes(b"%PDF-1.4\nstale\n")

    output = root / "dist" / "submission.zip"
    payload = build_package(
        root=root,
        output=output,
        style_dir=style,
        git_commit="d" * 40,
        submission_ready=True,
    )

    assert payload["status"] == "PASS"
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
    assert "figures/generated/stale.pdf" not in names
    assert {
        f"figures/generated/{path.name}"
        for path in paths.values()
    } <= names


def test_submission_overleaf_rejects_figure_hash_drift(tmp_path):
    root, style = make_root(tmp_path)
    paths = _write_submission_manifests(root)
    paths["adaptive_tradeoff"].write_bytes(b"%PDF-1.4\ntampered\n")

    with pytest.raises(ValueError, match="hash differs"):
        build_package(
            root=root,
            output=root / "dist" / "bad-submission.zip",
            style_dir=style,
            git_commit="e" * 40,
            submission_ready=True,
        )


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
