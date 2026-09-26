import sys
from pathlib import Path

from scripts.verify_cpu import run_step


def test_run_step_records_pass_and_log(tmp_path):
    record = run_step(
        "hello",
        [sys.executable, "-c", "print('ok')"],
        tmp_path,
    )
    assert record["status"] == "PASS"
    assert record["returncode"] == 0
    assert Path(record["log"]).read_text(encoding="utf-8").strip() == "ok"


def test_run_step_records_failure(tmp_path):
    record = run_step(
        "bad",
        [sys.executable, "-c", "raise SystemExit(3)"],
        tmp_path,
    )
    assert record["status"] == "FAIL"
    assert record["returncode"] == 3
