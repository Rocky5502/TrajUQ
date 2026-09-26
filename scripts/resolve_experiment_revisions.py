#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


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


def collect_model_specs(primary: dict, include_diagnostic: bool = False) -> list[dict]:
    groups = ["primary"]
    if include_diagnostic:
        groups.append("diagnostic")

    specs: list[dict] = []
    seen_ids: set[str] = set()
    for group in groups:
        for config_path in primary.get("models", {}).get(group, []):
            path = Path(config_path)
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            model_id = str(cfg["model_id"])
            if model_id in seen_ids:
                raise ValueError(f"duplicate model_id in experiment config: {model_id}")
            seen_ids.add(model_id)
            specs.append(
                {
                    "group": group,
                    "adapter": str(cfg["adapter"]),
                    "model_id": model_id,
                    "requested_revision": cfg.get("revision"),
                    "template": str(path),
                    "config": cfg,
                }
            )
    return specs


def collect_dataset_specs(primary: dict, manifest: dict) -> list[dict]:
    definitions = manifest.get("datasets", {})
    specs: list[dict] = []
    seen_repo_ids: set[str] = set()
    for name in primary.get("datasets", []):
        if name not in definitions:
            raise ValueError(f"dataset missing from evaluation manifest: {name}")
        cfg = definitions[name]
        repo_id = str(cfg["hf_id"])
        if repo_id in seen_repo_ids:
            raise ValueError(f"duplicate dataset repo id in primary suite: {repo_id}")
        seen_repo_ids.add(repo_id)
        specs.append(
            {
                "name": str(name),
                "repo_id": repo_id,
                "requested_revision": cfg.get("revision"),
                "gated": bool(cfg.get("gated", False)),
            }
        )
    return specs


def write_frozen_model_config(spec: dict, resolved_sha: str, root: Path) -> Path:
    cfg = dict(spec["config"])
    cfg["revision"] = resolved_sha
    path = root / f"{spec['adapter']}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve all model and dataset repositories in the frozen TrajUQ "
            "experiment to immutable Hugging Face commit SHAs."
        )
    )
    parser.add_argument(
        "--primary-config",
        default="configs/experiments/primary.yaml",
    )
    parser.add_argument(
        "--dataset-manifest",
        default="data/manifests/evaluation_splits.json",
    )
    parser.add_argument(
        "--include-diagnostic",
        action="store_true",
        help="Also resolve diagnostic-only model configs such as FLARE.",
    )
    parser.add_argument(
        "--frozen-model-dir",
        default="results/frozen_models",
    )
    parser.add_argument(
        "--output",
        default="results/revision_ledger.json",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit('Install model extras first: pip install -e ".[models]"') from exc

    primary_path = Path(args.primary_config)
    manifest_path = Path(args.dataset_manifest)
    primary = yaml.safe_load(primary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    model_specs = collect_model_specs(
        primary,
        include_diagnostic=args.include_diagnostic,
    )
    dataset_specs = collect_dataset_specs(primary, manifest)

    api = HfApi()
    failures: list[str] = []
    model_results: list[dict] = []
    dataset_results: list[dict] = []
    commit = git_head()
    if not commit:
        failures.append("git commit could not be resolved")

    for spec in model_specs:
        try:
            info = api.model_info(
                spec["model_id"],
                revision=spec["requested_revision"],
            )
            resolved_sha = getattr(info, "sha", None)
            if not resolved_sha:
                raise RuntimeError("Hugging Face returned no SHA")
            model_results.append(
                {
                    **{
                        key: spec[key]
                        for key in (
                            "group",
                            "adapter",
                            "model_id",
                            "requested_revision",
                            "template",
                        )
                    },
                    "resolved_sha": str(resolved_sha),
                }
            )
        except Exception as exc:
            failures.append(
                f"model {spec['model_id']}: {type(exc).__name__}: {exc}"
            )

    for spec in dataset_specs:
        try:
            info = api.dataset_info(
                spec["repo_id"],
                revision=spec["requested_revision"],
            )
            resolved_sha = getattr(info, "sha", None)
            if not resolved_sha:
                raise RuntimeError("Hugging Face returned no SHA")
            dataset_results.append(
                {
                    **spec,
                    "resolved_sha": str(resolved_sha),
                }
            )
        except Exception as exc:
            gated_note = " (gated; verify HF authentication)" if spec["gated"] else ""
            failures.append(
                f"dataset {spec['name']} / {spec['repo_id']}{gated_note}: "
                f"{type(exc).__name__}: {exc}"
            )

    status = "PASS" if not failures else "FAIL"
    payload = {
        "protocol": "trajuq_revision_ledger_v2",
        "status": status,
        "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "primary_config": str(primary_path),
        "primary_config_sha256": sha256_file(primary_path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "include_diagnostic": bool(args.include_diagnostic),
        "models": model_results,
        "datasets": dataset_results,
        "failures": failures,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if failures:
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(
            "Revision resolution failed. No frozen model configs were written."
        )

    frozen_root = Path(args.frozen_model_dir)
    by_adapter = {spec["adapter"]: spec for spec in model_specs}
    frozen_configs: dict[str, str] = {}
    frozen_hashes: dict[str, str] = {}
    for result in model_results:
        source = by_adapter[result["adapter"]]
        frozen_path = write_frozen_model_config(
            source,
            result["resolved_sha"],
            frozen_root,
        )
        frozen_configs[result["adapter"]] = str(frozen_path)
        frozen_hashes[result["adapter"]] = sha256_file(frozen_path)

    payload["frozen_model_configs"] = frozen_configs
    payload["frozen_model_config_sha256"] = frozen_hashes
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
