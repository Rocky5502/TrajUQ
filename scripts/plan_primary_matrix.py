#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


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
    readiness_path = Path(args.readiness)
    if not readiness_path.exists():
        failures.append(f"missing readiness artifact: {readiness_path}")
    else:
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        if readiness.get("status") != "PASS":
            failures.append("primary readiness gate is not PASS")

    primary_path = Path(args.primary_config)
    if not primary_path.exists():
        failures.append(f"missing primary config: {primary_path}")
        raise SystemExit("\n".join(failures))
    primary = load_yaml(primary_path)

    configured = dict(DEFAULT_FROZEN_MODELS)
    configured.update(dict(args.model_config))

    model_cfgs: dict[str, dict] = {}
    expected_adapters: list[str] = []
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
        model_cfgs[adapter] = frozen

    datasets = [str(x) for x in primary.get("datasets", [])]
    data_meta: dict[str, dict] = {}
    data_root = Path(args.data_root)
    for dataset in datasets:
        root = data_root / dataset
        cal = root / "calibration.jsonl"
        test = root / "test.jsonl"
        manifest = root / "split_manifest.json"
        for path in (cal, test, manifest):
            if not path.exists():
                failures.append(f"{dataset}: missing {path}")
        if manifest.exists():
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            if not meta.get("requested_revision"):
                failures.append(f"{dataset}: dataset revision is not pinned")
            if int(meta.get("n_calibration", 0)) <= 0:
                failures.append(f"{dataset}: empty calibration partition")
            if int(meta.get("n_test", 0)) <= 0:
                failures.append(f"{dataset}: empty test partition")
            data_meta[dataset] = meta

    if failures:
        payload = {
            "protocol": "trajuq_primary_matrix_plan_v1",
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
                    "dataset": dataset,
                    "dataset_revision": data_meta[dataset]["requested_revision"],
                    "calibration": partition_commands["calibration"],
                    "test": partition_commands["test"],
                }
            )

    payload = {
        "protocol": "trajuq_primary_matrix_plan_v1",
        "status": "PASS",
        "primary_config": str(primary_path),
        "readiness": str(readiness_path),
        "seed": seed,
        "models": expected_adapters,
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
