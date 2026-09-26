import subprocess
import sys
from pathlib import Path


def test_paper_audit_passes_in_pre_results_mode():
    assert Path("paper/main.tex").exists()
    subprocess.run(
        [sys.executable, "scripts/audit_paper.py"],
        check=True,
    )
