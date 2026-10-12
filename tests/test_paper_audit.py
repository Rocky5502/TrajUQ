import subprocess
import sys
from pathlib import Path


def test_paper_audit_passes_in_pre_results_mode():
    assert Path("paper/main.tex").exists()
    subprocess.run(
        [sys.executable, "scripts/audit_paper.py"],
        check=True,
    )

def _write_minimal_paper_tree(root: Path, *, extra_appendix: str = "") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.tex").write_text(
        "\\usepackage{tmlr}\n"
        "\\title{When Diffusion Language Models Are Unsure:\\\\\n"
        "Trajectory-Aware Uncertainty}\n"
        "\\author{\\name Anonymous Authors}\n"
        "\\begin{document}\n"
        "\\section{Conclusion}\n"
        "\\section*{Limitations}\n"
        "\\bibliographystyle{tmlr}\n"
        "\\bibliography{references}\n"
        "\\input{appendix}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    (root / "appendix.tex").write_text(extra_appendix, encoding="utf-8")
    (root / "references.bib").write_text("", encoding="utf-8")
    (root / "generated_results.tex").write_text("", encoding="utf-8")


def test_paper_audit_does_not_mistake_latex_linebreak_for_windows_path(tmp_path):
    paper = tmp_path / "paper"
    _write_minimal_paper_tree(paper)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_paper.py",
            "--paper-dir",
            str(paper),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout


def test_paper_audit_still_rejects_real_windows_absolute_path(tmp_path):
    paper = tmp_path / "paper"
    _write_minimal_paper_tree(
        paper,
        extra_appendix="Private workspace: G:\\\\NAACL Project2\\n",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_paper.py",
            "--paper-dir",
            str(paper),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode != 0
    assert "Windows absolute path" in completed.stdout

