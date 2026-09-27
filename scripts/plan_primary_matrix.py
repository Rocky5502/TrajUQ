#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from trajuq.data_integrity import audit_split_pair


DEFAULT_FROZEN_MODELS = {
    "llada": "results/frozen_models/llada.yaml",
    "sdar": "results/frozen_models/sdar.yaml",
    "dream": "results/frozen_models/dream.yaml",
}


def parse_override(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model-config must use ADAPTER=PATH")
    adapter, path = value.split("=", 1)
    adapter, path = adapter.strip(), path.strip()
    if not adapter or not path:
        raise argparse.ArgumentTypeError("ADAPTER and PATH must be non-empty")
    return adapter, path


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def git_status_porcelain() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def validate_readiness_source_bindings(readiness: dict) -> list[str]:
    failures: list[str] = []
    bindings = readiness.get("source_bindings")
    if not isinstance(bindings, list) or not bindings:
        return ["primary readiness has no source-binding manifest"]

    seen: set[str] = set()
    for item in bindings:
        if not isinstance(item, dict):
            failures.append("primary readiness source binding is malformed")
            continue
        path_text = str(item.get("path", "")).strip()
        expected_sha = str(item.get("sha256", "")).strip()
        if not path_text or not expected_sha:
            failures.append("primary readiness source binding lacks path/SHA-256")
            continue
        if path_text in seen:
            failures.append(
                f"primary readiness source binding duplicates path: {path_text}"
            )
            continue
        seen.add(path_text)
        path = Path(path_text)
        if not path.is_file():
            failures.append(
                f"primary readiness bound source is missing: {path_text}"
            )
            continue
        if sha256_file(path) != expected_sha:
            failures.append(
                f"primary readiness bound source hash drifted: {path_text}"
            )
    return failures


def generation_args(cfg: dict) -> list[str]:
    adapter = str(cfg["adapter"])
    g = cfg.get("generation", {})
    args: list[str] = []

    if adapter == "llada":
        args += [
            "--steps", str(g["steps"]),
            "--max-new-tokens", str(g["gen_length"]),
            "--block-length", str(g["block_length"]),
            "--temperature", str(g.get("temperature", 0.0)),
            "--llada-cfg-scale", str(g.get("cfg_scale", 0.0)),
            "--llada-remasking", str(g.get("remasking", "low_confidence")),
        ]
    elif adapter == "dream":
        args += [
            "--steps", str(g["steps"]),
            "--max-new-tokens", str(g["max_new_tokens"]),
            "--temperature", str(g.get("temperature", 0.0)),
            "--top-p", str(g.get("top_p", 1.0)),
            "--dream-alg", str(g.get("alg", "entropy")),
            "--dream-alg-temp", str(g.get("alg_temp", 0.1)),
        ]
    elif adapter == "sdar":
        args += [
            "--max-new-tokens", str(g["gen_length"]),
            "--block-length", str(g["block_length"]),
            "--temperature", str(g.get("temperature", 1.0)),
            "--top-p", str(g.get("top_p", 1.0)),
            "--sdar-denoising-steps", str(g["denoising_steps"]),
            "--sdar-top-k", str(g.get("top_k", 0)),
            "--sdar-remasking-strategy",
            str(g.get("remasking_strategy", "low_confidence_dynamic")),
            "--sdar-confidence-threshold",
            str(g.get("confidence_threshold", 0.85)),
        ]
    else:
        raise ValueError(f"Unsupported primary adapter: {adapter}")
    return args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the frozen primary model-dataset execution plan."
    )
    parser.add_argument(
        "--primary-config",
        default="configs/experiments/primary.yaml",
    )
    parser.add_argument(
        "--readiness",
        default="results/primary_readiness.json",
    )
    parser.add_argument(
        "--model-config",
        action="append",
        type=parse_override,
        default=[],
        help="Override ADAPTER=PATH; may be repeated.",
    )
    parser.add_argument(
        "--data-root",
        default="data/processed",
    )
    parser.add_argument(
        "--run-root",
        default="runs/primary",
    )
    parser.add_argument(
        "--output",
        default="results/primary_matrix_plan.json",
    )
    args = parser.parse_args()

    failures: list[str] = []
    readiness: dict = {}
    readiness_path = Path(args.readiness)
    if not readiness_path.exists():
        failures.append(f"missing readiness artifact: {readiness_path}")
    else:
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        if readiness.get("protocol") != "trajuq_primary_readiness_v2":
            failures.append("primary readiness protocol is not v2")
        if readiness.get("status") != "PASS":
            failures.append("primary readiness gate is not PASS")
        failures.extend(validate_readiness_source_bindings(readiness))
        current_head = git_head()
        experiment_commit = readiness.get("experiment_git_commit")
        if not current_head:
            failures.append("current git commit could not be resolved")
        elif current_head != experiment_commit:
            failures.append(
                "current git commit differs from the frozen experiment commit; "
                "do not plan primary execution from modified code"
            )
        current_status = git_status_porcelain()
        if current_status is None:
            failures.append("current git worktree status could not be resolved")
        elif current_status:
            failures.append(
                "current git worktree is dirty; do not plan primary execution "
                "from uncommitted code"
            )

    revision_ledger: dict = {}
    revision_ledger_path: Path | None = None
    if readiness:
        ledger_check = readiness.get("checks", {}).get("revision_ledger", {})
        ledger_path_text = str(ledger_check.get("path", "")).strip()
        if not ledger_path_text:
            failures.append("primary readiness does not identify its revision ledger")
        else:
            revision_ledger_path = Path(ledger_path_text)
            if not revision_ledger_path.exists():
                failures.append(
                    f"missing revision ledger bound by readiness: {revision_ledger_path}"
                )
            else:
                try:
                    revision_ledger = json.loads(
                        revision_ledger_path.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError as exc:
                    failures.append(f"invalid revision ledger JSON: {exc}")
                else:
                    if revision_ledger.get("protocol") != "trajuq_revision_ledger_v2":
                        failures.append("revision ledger protocol is not v2")
                    if revision_ledger.get("status") != "PASS":
                        failures.append("revision ledger is not PASS")
                    if (
                        revision_ledger.get("git_commit")
                        != readiness.get("experiment_git_commit")
                    ):
                        failures.append(
                            "revision ledger commit differs from primary readiness"
                        )

    primary_path = Path(args.primary_config)
    if not primary_path.exists():
        failures.append(f"missing primary config: {primary_path}")
        raise SystemExit("\n".join(failures))
    primary = load_yaml(primary_path)

    configured = dict(DEFAULT_FROZEN_MODELS)
    configured.update(dict(args.model_config))

    model_cfgs: dict[str, dict] = {}
    model_cfg_hashes: dict[str, str] = {}
    expected_adapters: list[str] = []
    ledger_models = {
        str(item.get("adapter")): item
        for item in revision_ledger.get("models", [])
        if item.get("adapter")
    }
    ledger_datasets = {
        str(item.get("name")): item
        for item in revision_ledger.get("datasets", [])
        if item.get("name")
    }
    for model_path in primary.get("models", {}).get("primary", []):
        template = load_yaml(Path(model_path))
        adapter = str(template.get("adapter", ""))
        expected_adapters.append(adapter)
        frozen_path = Path(configured.get(adapter, ""))
        if not frozen_path.exists():
            failures.append(
                f"{adapter}: missing frozen model config {frozen_path}"
            )
            continue
        frozen = load_yaml(frozen_path)
        revision = frozen.get("revision")
        if not revision or str(revision).lower() == "null":
            failures.append(f"{adapter}: frozen model config has no revision")
        if frozen.get("model_id") != template.get("model_id"):
            failures.append(f"{adapter}: frozen model_id differs from template")
        if frozen.get("adapter") != adapter:
            failures.append(f"{adapter}: frozen adapter differs from template")
        expected_model_revision = ledger_models.get(adapter, {}).get("resolved_sha")
        if not expected_model_revision:
            failures.append(f"{adapter}: missing resolved model SHA in revision ledger")
        elif str(revision) != str(expected_model_revision):
            failures.append(
                f"{adapter}: frozen model revision differs from revision ledger"
            )

        observed_model_config_hash = sha256_file(frozen_path)
        expected_model_config_hash = str(
            revision_ledger.get("frozen_model_config_sha256", {}).get(adapter, "")
        )
        expected_model_config_path = str(
            revision_ledger.get("frozen_model_configs", {}).get(adapter, "")
        )
        if not expected_model_config_hash:
            failures.append(
                f"{adapter}: revision ledger has no frozen model-config hash"
            )
        elif observed_model_config_hash != expected_model_config_hash:
            failures.append(
                f"{adapter}: frozen model config hash differs from revision ledger"
            )
        if expected_model_config_path and str(frozen_path) != expected_model_config_path:
            failures.append(
                f"{adapter}: frozen model config path differs from revision ledger"
            )

        model_cfgs[adapter] = frozen
        model_cfg_hashes[adapter] = observed_model_config_hash

    datasets = [str(x) for x in primary.get("datasets", [])]
    data_meta: dict[str, dict] = {}
    data_root = Path(args.data_root)
    for dataset in datasets:
        root = data_root / dataset
        cal = root / "calibration.jsonl"
        test = root / "test.jsonl"
        manifest = root / "split_manifest.json"
        split_audit = root / "split_audit.json"
        missing = [
            path
            for path in (cal, test, manifest, split_audit)
            if not path.exists()
        ]
        for path in missing:
            failures.append(f"{dataset}: missing {path}")
        if missing:
            continue

        try:
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            saved_audit = json.loads(split_audit.read_text(encoding="utf-8"))
            live_audit = audit_split_pair(cal, test, manifest)
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            failures.append(f"{dataset}: split artifacts are invalid: {exc}")
            continue

        if saved_audit != live_audit:
            failures.append(
                f"{dataset}: saved split audit is stale relative to current "
                "calibration/test/manifest artifacts"
            )
        if live_audit.get("protocol") != "trajuq_split_pair_audit_v1":
            failures.append(f"{dataset}: split audit protocol is not v1")
        if live_audit.get("status") != "PASS":
            failures.append(f"{dataset}: live split integrity audit is not PASS")
        if int(live_audit.get("id_overlap_count", -1)) != 0:
            failures.append(f"{dataset}: split audit reports ID overlap")
        if int(live_audit.get("exact_prompt_overlap_count", -1)) != 0:
            failures.append(f"{dataset}: split audit reports exact prompt overlap")

        requested_revision = meta.get("requested_revision")
        expected_dataset_revision = ledger_datasets.get(dataset, {}).get(
            "resolved_sha"
        )
        if not requested_revision:
            failures.append(f"{dataset}: dataset revision is not pinned")
        if not expected_dataset_revision:
            failures.append(
                f"{dataset}: missing resolved dataset SHA in revision ledger"
            )
        elif str(requested_revision) != str(expected_dataset_revision):
            failures.append(
                f"{dataset}: prepared split revision differs from revision ledger"
            )
        if int(meta.get("n_calibration", 0)) <= 0:
            failures.append(f"{dataset}: empty calibration partition")
        if int(meta.get("n_test", 0)) <= 0:
            failures.append(f"{dataset}: empty test partition")

        data_meta[dataset] = {
            "manifest": meta,
            "manifest_path": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "audit_path": str(split_audit),
            "audit_sha256": sha256_file(split_audit),
        }

    if failures:
        payload = {
            "protocol": "trajuq_primary_matrix_plan_v2",
            "status": "FAIL",
            "failures": failures,
            "cells": [],
        }
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(1)

    seed = int(primary.get("experiment", {}).get("seed", 2027))
    cells: list[dict] = []
    for adapter in expected_adapters:
        cfg = model_cfgs[adapter]
        for dataset in datasets:
            base = Path(args.run_root) / adapter / dataset
            partition_commands: dict[str, dict] = {}
            for partition in ("calibration", "test"):
                input_path = data_root / dataset / f"{partition}.jsonl"
                raw = base / f"{partition}.raw.jsonl"
                scored = base / f"{partition}.scored.jsonl"
                generate = [
                    "python",
                    "scripts/run_generation.py",
                    "--adapter", adapter,
                    "--model", str(cfg["model_id"]),
                    "--revision", str(cfg["revision"]),
                    "--input", str(input_path),
                    "--output", str(raw),
                    "--seed", str(seed),
                    *generation_args(cfg),
                ]
                audit_raw = [
                    "python", "scripts/audit_run.py",
                    "--run", str(raw),
                    "--stage", "raw",
                    "--require-revision",
                ]
                score = [
                    "python", "scripts/score_outputs.py",
                    "--run", str(raw),
                    "--input", str(input_path),
                    "--output", str(scored),
                ]
                audit_scored = [
                    "python", "scripts/audit_run.py",
                    "--run", str(scored),
                    "--stage", "scored",
                    "--require-revision",
                ]
                partition_commands[partition] = {
                    "input": str(input_path),
                    "input_sha256": sha256_file(input_path),
                    "raw_output": str(raw),
                    "scored_output": str(scored),
                    "generate_argv": generate,
                    "audit_raw_argv": audit_raw,
                    "score_argv": score,
                    "audit_scored_argv": audit_scored,
                }

            cells.append(
                {
                    "cell": f"{adapter}:{dataset}",
                    "adapter": adapter,
                    "model_id": cfg["model_id"],
                    "model_revision": cfg["revision"],
                    "model_config_sha256": model_cfg_hashes[adapter],
                    "dataset": dataset,
                    "dataset_revision": data_meta[dataset]["manifest"]["requested_revision"],
                    "split_manifest": data_meta[dataset]["manifest_path"],
                    "split_manifest_sha256": data_meta[dataset]["manifest_sha256"],
                    "split_audit": data_meta[dataset]["audit_path"],
                    "split_audit_sha256": data_meta[dataset]["audit_sha256"],
                    "calibration": partition_commands["calibration"],
                    "test": partition_commands["test"],
                }
            )

    payload = {
        "protocol": "trajuq_primary_matrix_plan_v2",
        "status": "PASS",
        "git_commit": git_head(),
        "experiment_git_commit": readiness.get("experiment_git_commit"),
        "primary_config": str(primary_path),
        "primary_config_sha256": sha256_file(primary_path),
        "readiness": str(readiness_path),
        "readiness_sha256": sha256_file(readiness_path),
        "revision_ledger": (
            str(revision_ledger_path)
            if revision_ledger_path is not None
            else None
        ),
        "revision_ledger_sha256": (
            sha256_file(revision_ledger_path)
            if revision_ledger_path is not None
            else None
        ),
        "seed": seed,
        "models": expected_adapters,
        "model_configs": {
            adapter: {
                "path": str(Path(configured[adapter])),
                "sha256": model_cfg_hashes[adapter],
                "revision": model_cfgs[adapter]["revision"],
            }
            for adapter in expected_adapters
        },
        "datasets": datasets,
        "expected_cells": len(expected_adapters) * len(datasets),
        "cell_count": len(cells),
        "execution_policy": (
            "Plan only. Execute cells without changing frozen configs; "
            "never tune on test outcomes."
        ),
        "cells": cells,
        "failures": [],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(out),
                "cell_count": len(cells),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
