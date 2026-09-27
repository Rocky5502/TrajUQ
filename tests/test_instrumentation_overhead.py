import hashlib
import json
import sys
from pathlib import Path

import pytest

import scripts.compare_instrumentation_overhead as overhead


def write_run(path: Path, *, time_offset: float = 0.0) -> None:
    rows = []
    for index, item_id in enumerate(("a", "b")):
        rows.append(
            {
                "example_id": item_id,
                "dataset": "gsm8k",
                "model": "example/model",
                "adapter": "llada",
                "seed": 2027,
                "metadata": {
                    "source_git_commit": "frozen-commit",
                    "source_git_clean": True,
                },
                "generation": {
                    "metadata": {
                        "model_revision": "model-sha",
                        "wall_time_seconds": 1.0 + index + time_offset,
                        "peak_gpu_memory_bytes": 1000 + index,
                    }
                },
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_main(
    monkeypatch,
    reference: Path,
    instrumented: Path,
    output: Path,
) -> None:
    monkeypatch.setattr(overhead, "git_head", lambda: "frozen-commit")
    monkeypatch.setattr(overhead, "git_clean", lambda: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_instrumentation_overhead.py",
            "--reference",
            str(reference),
            "--instrumented",
            str(instrumented),
            "--output",
            str(output),
            "--n-boot",
            "20",
            "--seed",
            "2027",
        ],
    )
    overhead.main()


def test_overhead_artifact_is_source_bound_and_immutable(tmp_path, monkeypatch):
    reference = tmp_path / "reference.jsonl"
    instrumented = tmp_path / "instrumented.jsonl"
    output = tmp_path / "overhead.json"
    write_run(reference)
    write_run(instrumented, time_offset=0.25)

    run_main(monkeypatch, reference, instrumented, output)
    first = output.read_bytes()
    payload = json.loads(first)

    assert payload["protocol"] == "trajuq_instrumentation_overhead_v2"
    assert payload["git_commit"] == "frozen-commit"
    assert payload["git_clean"] is True
    assert payload["reference_sha256"] == sha256(reference)
    assert payload["instrumented_sha256"] == sha256(instrumented)
    assert payload["run_provenance"]["model_revision"] == "model-sha"
    assert payload["run_provenance"]["source_git_commit"] == "frozen-commit"
    assert payload["run_provenance"]["source_git_clean"] is True

    run_main(monkeypatch, reference, instrumented, output)
    assert output.read_bytes() == first

    write_run(instrumented, time_offset=0.5)
    with pytest.raises(FileExistsError, match="Refusing to overwrite immutable JSON artifact"):
        run_main(monkeypatch, reference, instrumented, output)


def test_resume_wrappers_revalidate_overhead_artifacts():
    linux = Path("scripts/pilot_linux.sh").read_text(encoding="utf-8")
    remaining_linux = Path("scripts/parity_remaining_linux.sh").read_text(
        encoding="utf-8"
    )
    windows = Path("scripts/pilot_windows.ps1").read_text(encoding="utf-8")
    remaining_windows = Path(
        "scripts/parity_remaining_windows.ps1"
    ).read_text(encoding="utf-8")

    assert 'if [[ ! -e "$PARITY_OVERHEAD" ]]' not in linux
    assert 'if [[ ! -e "$overhead" ]]' not in remaining_linux
    assert "-not (Test-Path $ParityOverhead)" not in windows
    assert "-not (Test-Path $Overhead)" not in remaining_windows


def test_overhead_rejects_sources_from_other_commit(tmp_path, monkeypatch):
    reference = tmp_path / "reference.jsonl"
    instrumented = tmp_path / "instrumented.jsonl"
    output = tmp_path / "overhead.json"
    write_run(reference)
    write_run(instrumented, time_offset=0.25)

    rows = [
        json.loads(line)
        for line in reference.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        row["metadata"]["source_git_commit"] = "old-commit"
    reference.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    rows = [
        json.loads(line)
        for line in instrumented.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        row["metadata"]["source_git_commit"] = "old-commit"
    instrumented.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="different git commit"):
        run_main(monkeypatch, reference, instrumented, output)


def test_overhead_rejects_dirty_source_run(tmp_path, monkeypatch):
    reference = tmp_path / "reference.jsonl"
    instrumented = tmp_path / "instrumented.jsonl"
    output = tmp_path / "overhead.json"
    write_run(reference)
    write_run(instrumented, time_offset=0.25)

    rows = [
        json.loads(line)
        for line in instrumented.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        row["metadata"]["source_git_clean"] = False
    instrumented.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="run provenance differs"):
        run_main(monkeypatch, reference, instrumented, output)
