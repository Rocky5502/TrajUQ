import json
import subprocess
import sys
from pathlib import Path


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_compare_run_parity_passes(tmp_path):
    rows = [
        {
            "example_id": "a",
            "generation": {"text": "x", "token_ids": [1, 2, 3]},
        },
        {
            "example_id": "b",
            "generation": {"text": "y", "token_ids": [4, 5]},
        },
    ]
    ref = tmp_path / "ref.jsonl"
    inst = tmp_path / "inst.jsonl"
    write(ref, rows)
    write(inst, rows)
    subprocess.run(
        [
            sys.executable,
            "scripts/compare_run_parity.py",
            "--reference",
            str(ref),
            "--instrumented",
            str(inst),
            "--require-token-ids",
        ],
        check=True,
    )
