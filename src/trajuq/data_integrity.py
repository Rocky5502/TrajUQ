from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .datasets import NORMALIZER_VERSION, SPLIT_SELECTION_VERSION
from .io import read_jsonl


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def audit_split_pair(
    calibration_path: str | Path,
    test_path: str | Path,
    manifest_path: str | Path,
) -> dict:
    cal_path = Path(calibration_path)
    test_path = Path(test_path)
    meta_path = Path(manifest_path)

    failures: list[str] = []
    warnings: list[str] = []

    for path in (cal_path, test_path, meta_path):
        if not path.exists():
            failures.append(f"missing required split artifact: {path}")

    if failures:
        return {
            "status": "FAIL",
            "failures": failures,
            "warnings": warnings,
        }

    manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    cal = list(read_jsonl(cal_path))
    test = list(read_jsonl(test_path))

    def inspect(rows: list[dict], role: str) -> tuple[list[str], list[str], list[str]]:
        ids: list[str] = []
        prompts: list[str] = []
        datasets: list[str] = []
        for i, row in enumerate(rows):
            item_id = str(row.get("id", ""))
            prompt = row.get("prompt")
            reference = row.get("reference")
            dataset = str(row.get("dataset", ""))
            if not item_id:
                failures.append(f"{role} row {i}: missing id")
            if not isinstance(prompt, str) or not prompt.strip():
                failures.append(f"{role} row {i}: missing prompt")
                prompt = ""
            if reference is None or not str(reference).strip():
                failures.append(f"{role} row {i}: missing reference")
            if not dataset:
                failures.append(f"{role} row {i}: missing dataset")
            ids.append(item_id)
            prompts.append(prompt_sha(prompt))
            datasets.append(dataset)

        if len(ids) != len(set(ids)):
            failures.append(f"{role}: duplicate example IDs")
        duplicate_prompt_count = len(prompts) - len(set(prompts))
        if duplicate_prompt_count:
            warnings.append(
                f"{role}: {duplicate_prompt_count} duplicate prompt occurrences within partition"
            )
        return ids, prompts, datasets

    cal_ids, cal_prompts, cal_datasets = inspect(cal, "calibration")
    test_ids, test_prompts, test_datasets = inspect(test, "test")

    id_overlap = sorted(set(cal_ids).intersection(test_ids))
    prompt_overlap = sorted(set(cal_prompts).intersection(test_prompts))
    if id_overlap:
        failures.append(f"calibration/test ID overlap: {len(id_overlap)}")
    if prompt_overlap:
        failures.append(
            f"calibration/test exact-prompt overlap: {len(prompt_overlap)}"
        )

    declared_dataset = str(manifest.get("dataset", ""))
    observed_datasets = set(cal_datasets + test_datasets)
    if observed_datasets != {declared_dataset}:
        failures.append(
            f"dataset identity mismatch: manifest={declared_dataset!r}, "
            f"observed={sorted(observed_datasets)}"
        )

    if manifest.get("normalizer_version") != NORMALIZER_VERSION:
        failures.append(
            f"normalizer_version={manifest.get('normalizer_version')!r}; "
            f"expected {NORMALIZER_VERSION!r}"
        )
    if manifest.get("selection_version") != SPLIT_SELECTION_VERSION:
        failures.append(
            f"selection_version={manifest.get('selection_version')!r}; "
            f"expected {SPLIT_SELECTION_VERSION!r}"
        )

    cap_spec = manifest.get("pre_generation_cap")
    if not isinstance(cap_spec, dict):
        failures.append("split manifest has no pre_generation_cap object")
        cap_spec = {}
    elif cap_spec.get("method") != SPLIT_SELECTION_VERSION:
        failures.append("split manifest cap method is not the frozen hash-rank method")
    if not str(cap_spec.get("seed", "")).strip():
        failures.append("split manifest cap seed is missing")

    source_counts = manifest.get("source_counts")
    if not isinstance(source_counts, dict):
        failures.append("split manifest has no source_counts object")
        source_counts = {}

    parsed_source_counts: dict[str, int] = {}
    for key in (
        "source_total",
        "calibration_before_cap",
        "test_before_cap",
    ):
        value = source_counts.get(key)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            failures.append(f"invalid source count: {key}")
            continue
        if parsed < 0:
            failures.append(f"source count must be non-negative: {key}")
            continue
        parsed_source_counts[key] = parsed

    if all(
        key in parsed_source_counts
        for key in (
            "source_total",
            "calibration_before_cap",
            "test_before_cap",
        )
    ):
        expected_source_total = (
            parsed_source_counts["calibration_before_cap"]
            + parsed_source_counts["test_before_cap"]
        )
        if parsed_source_counts["source_total"] != expected_source_total:
            failures.append(
                "source_total differs from calibration_before_cap + "
                "test_before_cap"
            )

    for role, rows, cap_key, source_key in (
        ("calibration", cal, "calibration_max", "calibration_before_cap"),
        ("test", test, "test_max", "test_before_cap"),
    ):
        source_count = parsed_source_counts.get(source_key)
        if source_count is None:
            failures.append(f"{role}: invalid source count before cap")
            continue
        if source_count < len(rows):
            failures.append(f"{role}: source count before cap is smaller than output")
        cap = cap_spec.get(cap_key)
        if cap is None:
            expected_count = source_count
        else:
            try:
                cap = int(cap)
            except (TypeError, ValueError):
                failures.append(f"{role}: invalid frozen cap")
                continue
            if cap <= 0:
                failures.append(f"{role}: frozen cap must be positive")
                continue
            expected_count = min(source_count, cap)
        if len(rows) != expected_count:
            failures.append(
                f"{role}: prepared count {len(rows)} differs from frozen cap "
                f"expectation {expected_count}"
            )
    if not manifest.get("requested_revision"):
        failures.append("dataset revision is not pinned in split manifest")
    if not str(manifest.get("source_git_commit", "")).strip():
        failures.append("prepared split source git commit is missing")
    if manifest.get("source_git_clean") is not True:
        failures.append("prepared split was not produced from a clean git checkout")

    if int(manifest.get("n_calibration", -1)) != len(cal):
        failures.append("calibration count differs from split manifest")
    if int(manifest.get("n_test", -1)) != len(test):
        failures.append("test count differs from split manifest")
    if manifest.get("calibration_sha256") != file_sha256(cal_path):
        failures.append("calibration SHA-256 differs from split manifest")
    if manifest.get("test_sha256") != file_sha256(test_path):
        failures.append("test SHA-256 differs from split manifest")

    return {
        "protocol": "trajuq_split_pair_audit_v1",
        "status": "PASS" if not failures else "FAIL",
        "dataset": declared_dataset,
        "requested_revision": manifest.get("requested_revision"),
        "source_git_commit": manifest.get("source_git_commit"),
        "source_git_clean": manifest.get("source_git_clean"),
        "normalizer_version": manifest.get("normalizer_version"),
        "selection_version": manifest.get("selection_version"),
        "pre_generation_cap": manifest.get("pre_generation_cap"),
        "source_counts": manifest.get("source_counts"),
        "calibration_records": len(cal),
        "test_records": len(test),
        "id_overlap_count": len(id_overlap),
        "exact_prompt_overlap_count": len(prompt_overlap),
        "calibration_sha256": file_sha256(cal_path),
        "test_sha256": file_sha256(test_path),
        "failures": failures,
        "warnings": warnings,
    }
