from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


LINUX_WRAPPERS = (
    "scripts/freeze_preprimary.sh",
    "scripts/verify_cpu.sh",
    "scripts/pilot_linux.sh",
    "scripts/parity_remaining_linux.sh",
    "scripts/math500_gate_linux.sh",
    "scripts/a100_campaign_linux.sh",
    "scripts/a100_72h_autopilot.sh",
)


@pytest.mark.parametrize("script", LINUX_WRAPPERS)
def test_linux_wrapper_parses_with_bash(script: str):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable on this test host")

    path = Path(script)
    assert path.is_file(), f"missing Linux wrapper: {script}"

    # Windows Git checkouts may expose CRLF even when the committed shell
    # scripts are intended for Linux. Syntax-check normalized committed text
    # through stdin so this cross-platform test validates Bash grammar rather
    # than the host checkout's newline conversion. .gitattributes separately
    # forces *.sh to LF for future checkouts and Linux execution.
    source = path.read_text(encoding="utf-8")
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    completed = subprocess.run(
        [bash, "-n"],
        input=normalized,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0, (
        f"{script} failed Bash syntax validation:\n{completed.stdout}"
    )
