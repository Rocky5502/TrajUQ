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

    # Read raw bytes so Windows universal-newline translation cannot turn the
    # normalized LF stream back into CRLF while feeding Bash over stdin.
    # .gitattributes separately forces *.sh to LF in fresh Git checkouts.
    source = path.read_bytes()
    normalized = source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    completed = subprocess.run(
        [bash, "-n"],
        input=normalized,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = (completed.stdout or b"").decode("utf-8", errors="replace")
    assert completed.returncode == 0, (
        f"{script} failed Bash syntax validation:\n{output}"
    )
