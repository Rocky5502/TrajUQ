#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_retryable_remote_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None
    if status_code is not None:
        if status_code in {408, 409, 425, 429} or 500 <= status_code <= 599:
            return True
        if status_code in {400, 401, 403, 404}:
            return False

    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        any(
            token in name
            for token in (
                "connecterror",
                "connecttimeout",
                "readtimeout",
                "writetimeout",
                "proxyerror",
                "sslerror",
                "timeout",
            )
        )
        or any(
            token in message
            for token in (
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
        )
    )


def git_text(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the current experiment machine can actually read one "
            "record from every gated primary dataset at its frozen revision."
        )
    )
    parser.add_argument(
        "--revision-ledger",
        default="results/revision_ledger.json",
    )
    parser.add_argument(
        "--output",
        default="results/gated_dataset_access.json",
    )
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Explicit recovery-only replacement after inspecting/archiving an "
            "existing access artifact. Official freeze wrappers never use this."
        ),
    )
    args = parser.parse_args()
    if args.max_attempts < 1:
        raise SystemExit("--max-attempts must be >= 1")
    if args.retry_base_seconds < 0:
        raise SystemExit("--retry-base-seconds must be >= 0")

    out = Path(args.output)
    if out.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite gated dataset access evidence: {out}. "
            "Inspect/archive it before recovery."
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            'Install model/data extras first: pip install -e ".[models]"'
        ) from exc

    ledger_path = Path(args.revision_ledger)
    if not ledger_path.is_file():
        raise SystemExit(f"Missing revision ledger: {ledger_path}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    commit = git_text("rev-parse", "HEAD")
    status_text = git_text("status", "--porcelain")
    clean = None if status_text is None else status_text == ""

    if ledger.get("protocol") != "trajuq_revision_ledger_v2":
        failures.append("revision ledger protocol is not v2")
    if ledger.get("status") != "PASS":
        failures.append("revision ledger is not PASS")
    if not commit:
        failures.append("git commit could not be resolved")
    if clean is not True:
        failures.append("git worktree is not clean")
    if commit and ledger.get("git_commit") != commit:
        failures.append("revision ledger belongs to another git commit")

    manifest_path = Path(str(ledger.get("dataset_manifest", "")))
    if not manifest_path.is_file():
        failures.append("revision-ledger dataset manifest is missing")
        manifest: dict = {}
    else:
        observed_manifest_sha = sha256_file(manifest_path)
        if observed_manifest_sha != ledger.get("dataset_manifest_sha256"):
            failures.append("dataset manifest hash differs from revision ledger")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ledger_datasets = {
        str(item.get("name")): item
        for item in ledger.get("datasets", [])
        if item.get("name")
    }
    definitions = manifest.get("datasets", {})
    gated_names = sorted(
        name
        for name, spec in definitions.items()
        if spec.get("gated") is True and name in ledger_datasets
    )

    records: list[dict] = []
    if not failures:
        for name in gated_names:
            spec = definitions[name]
            ledger_item = ledger_datasets[name]
            revision = str(ledger_item.get("resolved_sha", ""))
            repo_id = str(ledger_item.get("repo_id", ""))
            split = str(
                spec.get("source_split")
                or spec.get("calibration_split")
                or spec.get("test_split")
                or ""
            )
            config = spec.get("config")

            record = {
                "name": name,
                "repo_id": repo_id,
                "config": config,
                "split": split,
                "resolved_sha": revision,
                "status": "FAIL",
            }
            if not revision or not repo_id or not split:
                record["failure"] = "frozen dataset metadata is incomplete"
                failures.append(f"{name}: frozen dataset metadata is incomplete")
                records.append(record)
                continue

            try:
                kwargs = {
                    "split": split,
                    "revision": revision,
                    "streaming": True,
                }
                last_exc: Exception | None = None
                for attempt in range(1, args.max_attempts + 1):
                    try:
                        if config:
                            dataset = load_dataset(
                                repo_id,
                                str(config),
                                **kwargs,
                            )
                        else:
                            dataset = load_dataset(repo_id, **kwargs)
                        first = next(iter(dataset))
                        if not isinstance(first, dict) or not first:
                            raise RuntimeError(
                                "first streamed record is empty or invalid"
                            )
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        if (
                            not is_retryable_remote_error(exc)
                            or attempt >= args.max_attempts
                        ):
                            raise
                        delay = min(
                            args.retry_base_seconds * (2 ** (attempt - 1)),
                            30.0,
                        )
                        print(
                            f"[TrajUQ] transient gated-data failure for {name}; "
                            f"retry {attempt + 1}/{args.max_attempts} in "
                            f"{delay:.1f}s ({type(exc).__name__})"
                        )
                        time.sleep(delay)
                if last_exc is not None:
                    raise last_exc
                record["status"] = "PASS"
                record["record_accessed"] = True
            except Exception as exc:
                # Do not persist raw upstream error text: authentication
                # failures can contain request details that do not belong in
                # a handoff artifact.
                record["failure_type"] = type(exc).__name__
                record["failure"] = "gated dataset record access failed"
                failures.append(
                    f"{name}: gated dataset record access failed "
                    f"({type(exc).__name__})"
                )
            records.append(record)

    payload = {
        "protocol": "trajuq_gated_dataset_access_v1",
        "status": "PASS" if not failures else "FAIL",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_clean": clean,
        "revision_ledger": str(ledger_path),
        "revision_ledger_sha256": sha256_file(ledger_path),
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": (
            sha256_file(manifest_path) if manifest_path.is_file() else None
        ),
        "gated_dataset_names": gated_names,
        "datasets": records,
        "failure_count": len(failures),
        "failures": failures,
        "privacy_note": (
            "No gated benchmark question, answer, or sample content is stored "
            "in this artifact."
        ),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
