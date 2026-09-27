#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from trajuq.datasets import NORMALIZER_VERSION, normalize_row
from trajuq.scoring import gsm8k_reference_number, score_record


MC_DATASETS = {"arc_challenge", "mmlu_pro", "truthfulqa", "gpqa_diamond"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def contract_split(spec: dict[str, Any]) -> str:
    strategy = str(spec.get("strategy", ""))
    if strategy == "official":
        return str(spec["calibration_split"])
    if strategy == "internal_holdout":
        return str(spec["source_split"])
    raise ValueError(f"unsupported split strategy: {strategy!r}")


def correct_smoke_output(dataset: str, reference: str) -> str:
    if dataset in MC_DATASETS:
        value = str(reference).strip().upper()
        if len(value) != 1 or not ("A" <= value <= "J"):
            raise ValueError("multiple-choice normalizer produced invalid reference")
        return f"FINAL: {value}"
    if dataset == "gsm8k":
        value = gsm8k_reference_number(str(reference))
        if value is None:
            raise ValueError("GSM8K reference has no parseable final number")
        rendered = str(int(value)) if float(value).is_integer() else format(value, ".12g")
        return f"FINAL: {rendered}"
    if dataset == "math500":
        value = str(reference).strip()
        if not value:
            raise ValueError("MATH-500 reference is empty")
        return f"FINAL: {value}"
    raise ValueError(f"unsupported primary dataset: {dataset}")


def verify_one(
    *,
    name: str,
    spec: dict[str, Any],
    revision: str,
    load_dataset: Callable[..., Any],
) -> dict[str, Any]:
    split = contract_split(spec)
    kwargs: dict[str, Any] = {
        "split": split,
        "revision": revision,
        "streaming": True,
    }
    config = spec.get("config")
    if config:
        stream = load_dataset(spec["hf_id"], config, **kwargs)
    else:
        stream = load_dataset(spec["hf_id"], **kwargs)

    row = dict(next(iter(stream)))
    normalized = normalize_row(name, row, 0)
    failures: list[str] = []

    if str(normalized.get("dataset", "")) != name:
        failures.append("normalized dataset identity mismatch")
    if not str(normalized.get("id", "")).strip():
        failures.append("normalized id is empty")
    if not str(normalized.get("prompt", "")).strip():
        failures.append("normalized prompt is empty")
    if normalized.get("reference") is None or not str(normalized["reference"]).strip():
        failures.append("normalized reference is empty")

    scorer = None
    scorer_smoke_pass = False
    if not failures:
        output = correct_smoke_output(name, str(normalized["reference"]))
        scorer_smoke_pass, scorer = score_record(
            name,
            output,
            str(normalized["reference"]),
        )
        if not scorer_smoke_pass:
            failures.append("frozen scorer rejected constructed correct answer")

    return {
        "dataset": name,
        "repo_id": str(spec["hf_id"]),
        "config": config,
        "split": split,
        "resolved_revision": revision,
        "normalizer_version": NORMALIZER_VERSION,
        "upstream_fields": sorted(str(key) for key in row),
        "normalized_contract": {
            "id_nonempty": bool(str(normalized.get("id", "")).strip()),
            "prompt_nonempty": bool(str(normalized.get("prompt", "")).strip()),
            "reference_nonempty": bool(
                normalized.get("reference") is not None
                and str(normalized.get("reference", "")).strip()
            ),
        },
        "scorer": scorer,
        "scorer_smoke_pass": bool(scorer_smoke_pass),
        "source_row_content_stored": False,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stream one record from every exact-revision primary dataset and "
            "verify the frozen normalizer/scorer contract without storing "
            "benchmark row content."
        )
    )
    parser.add_argument(
        "--revision-ledger",
        default="results/revision_ledger.json",
    )
    parser.add_argument(
        "--dataset-manifest",
        default="data/manifests/evaluation_splits.json",
    )
    parser.add_argument(
        "--output",
        default="results/dataset_contracts.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Explicit recovery-only replacement. Official freeze wrappers "
            "never use this."
        ),
    )
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite dataset-contract evidence: {out}. "
            "Inspect/archive it before recovery."
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit('Install model/data extras first: pip install -e ".[models]"') from exc

    ledger_path = Path(args.revision_ledger)
    manifest_path = Path(args.dataset_manifest)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    head = git_text("rev-parse", "HEAD")
    dirty_text = git_text("status", "--porcelain")
    git_clean = dirty_text == "" if dirty_text is not None else False

    if ledger.get("protocol") != "trajuq_revision_ledger_v2":
        failures.append("revision ledger protocol is not v2")
    if ledger.get("status") != "PASS":
        failures.append("revision ledger is not PASS")
    if not head or ledger.get("git_commit") != head:
        failures.append("revision ledger commit differs from current HEAD")
    if not git_clean:
        failures.append("git worktree is not clean")
    if ledger.get("dataset_manifest_sha256") != sha256_file(manifest_path):
        failures.append("dataset manifest hash differs from revision ledger")

    definitions = manifest.get("datasets", {})
    ledger_datasets = {
        str(item.get("name")): item
        for item in ledger.get("datasets", [])
        if item.get("name")
    }
    expected_names = [str(name) for name in definitions if name in ledger_datasets]
    if set(expected_names) != set(ledger_datasets):
        failures.append("revision ledger and dataset manifest primary entries disagree")

    results: list[dict[str, Any]] = []
    if not failures:
        for name in expected_names:
            spec = definitions[name]
            revision = str(ledger_datasets[name].get("resolved_sha", "")).strip()
            if not revision:
                failures.append(f"{name}: missing resolved dataset revision")
                continue
            try:
                result = verify_one(
                    name=name,
                    spec=spec,
                    revision=revision,
                    load_dataset=load_dataset,
                )
            except Exception as exc:
                result = {
                    "dataset": name,
                    "repo_id": str(spec.get("hf_id", "")),
                    "config": spec.get("config"),
                    "split": (
                        contract_split(spec)
                        if spec.get("strategy") in {"official", "internal_holdout"}
                        else None
                    ),
                    "resolved_revision": revision,
                    "normalizer_version": NORMALIZER_VERSION,
                    "source_row_content_stored": False,
                    "status": "FAIL",
                    "failures": [f"{type(exc).__name__}: dataset contract check failed"],
                }
            results.append(result)
            if result.get("status") != "PASS":
                failures.append(f"{name}: dataset normalizer/scorer contract failed")

    payload = {
        "protocol": "trajuq_dataset_contracts_v1",
        "status": "PASS" if not failures else "FAIL",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": head,
        "git_clean": git_clean,
        "revision_ledger": str(ledger_path),
        "revision_ledger_sha256": sha256_file(ledger_path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "normalizer_version": NORMALIZER_VERSION,
        "expected_datasets": sorted(ledger_datasets),
        "observed_datasets": sorted(
            str(item.get("dataset")) for item in results
        ),
        "results": results,
        "failure_count": len(failures),
        "failures": failures,
        "privacy_note": (
            "No streamed benchmark question, answer, prompt, or reference "
            "content is stored in this artifact."
        ),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "status": payload["status"],
                "git_commit": payload["git_commit"],
                "expected_datasets": payload["expected_datasets"],
                "observed_datasets": payload["observed_datasets"],
                "failure_count": payload["failure_count"],
                "output": str(out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
