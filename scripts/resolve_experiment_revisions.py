#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
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


def _http_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def is_retryable_remote_error(exc: Exception) -> bool:
    """Return True only for transport/rate/server failures worth retrying."""

    status = _http_status_code(exc)
    if status is not None:
        if status in {408, 409, 425, 429} or 500 <= status <= 599:
            return True
        if status in {400, 401, 403, 404}:
            return False

    name = type(exc).__name__.lower()
    retryable_names = (
        "connecterror",
        "connecttimeout",
        "readtimeout",
        "writetimeout",
        "pooltimeout",
        "proxyerror",
        "networkerror",
        "sslerror",
        "timeout",
    )
    if any(token in name for token in retryable_names):
        return True

    message = str(exc).lower()
    retryable_markers = (
        "unexpected_eof_while_reading",
        "unexpected eof",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "temporary failure",
        "timed out",
        "timeout",
        "rate limit",
        "too many requests",
        "server error",
    )
    return any(marker in message for marker in retryable_markers)


def resolve_remote_sha(
    fetch_info,
    *,
    label: str,
    max_attempts: int = 5,
    base_delay_seconds: float = 2.0,
    sleep_fn=time.sleep,
) -> str:
    """Resolve one Hub object SHA with bounded deterministic retry."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    if base_delay_seconds < 0:
        raise ValueError("base_delay_seconds must be >= 0")

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            info = fetch_info()
            resolved_sha = getattr(info, "sha", None)
            if not resolved_sha:
                raise RuntimeError("Hugging Face returned no SHA")
            return str(resolved_sha)
        except Exception as exc:
            last_exc = exc
            retryable = is_retryable_remote_error(exc)
            if not retryable or attempt >= max_attempts:
                raise
            delay = min(base_delay_seconds * (2 ** (attempt - 1)), 30.0)
            print(
                f"[TrajUQ] transient Hub failure while resolving {label}; "
                f"retry {attempt + 1}/{max_attempts} in {delay:.1f}s "
                f"({type(exc).__name__})"
            )
            sleep_fn(delay)

    assert last_exc is not None
    raise last_exc


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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Explicit recovery-only replacement of existing freeze artifacts. "
            "Never use after parity, pilot, or benchmark outcomes exist."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help=(
            "Maximum attempts for transient Hugging Face metadata failures. "
            "Authentication/not-found errors fail immediately."
        ),
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=2.0,
        help="Initial deterministic backoff between transient Hub retries.",
    )
    args = parser.parse_args()

    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")
    if args.retry_base_seconds < 0:
        raise SystemExit("--retry-base-seconds must be >= 0")

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

    out = Path(args.output)
    frozen_root = Path(args.frozen_model_dir)
    frozen_targets = [
        frozen_root / f"{spec['adapter']}.yaml"
        for spec in model_specs
    ]
    existing = [
        path
        for path in (out, *frozen_targets)
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise SystemExit(
            "Refusing to overwrite frozen revision evidence: "
            + ", ".join(str(path) for path in existing)
            + ". Inspect/archive it before recovery. Official freeze "
            "wrappers never overwrite frozen revision evidence."
        )

    api = HfApi()
    failures: list[str] = []
    model_results: list[dict] = []
    dataset_results: list[dict] = []
    commit = git_head()
    if not commit:
        failures.append("git commit could not be resolved")

    for spec in model_specs:
        try:
            resolved_sha = resolve_remote_sha(
                lambda spec=spec: api.model_info(
                    spec["model_id"],
                    revision=spec["requested_revision"],
                ),
                label=f"model {spec['model_id']}",
                max_attempts=args.max_attempts,
                base_delay_seconds=args.retry_base_seconds,
            )
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
            resolved_sha = resolve_remote_sha(
                lambda spec=spec: api.dataset_info(
                    spec["repo_id"],
                    revision=spec["requested_revision"],
                ),
                label=f"dataset {spec['name']} / {spec['repo_id']}",
                max_attempts=args.max_attempts,
                base_delay_seconds=args.retry_base_seconds,
            )
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
