from pathlib import Path

from scripts.source_only_manuscript_audit import audit


def test_source_only_audit_excludes_tests_and_tmp(tmp_path: Path) -> None:
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper/main.tex").write_text(
        r"\documentclass{article}\begin{document}clean\end{document}",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_noise.py").write_text("# TODO", encoding="utf-8")
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp/noise.tex").write_text("TBD", encoding="utf-8")

    result = audit(tmp_path)

    assert result["status"] == "PASS"
    assert result["counts"]["placeholder_markers"] == 0


def test_source_only_audit_finds_missing_citation(tmp_path: Path) -> None:
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper/main.tex").write_text(r"\citep{missing}", encoding="utf-8")
    (tmp_path / "paper/references.bib").write_text("", encoding="utf-8")

    result = audit(tmp_path)

    assert result["status"] == "BLOCKED"
    assert "missing" in result["missing_citations"]
