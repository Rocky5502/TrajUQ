#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import yaml


TOKENIZER_FILENAMES = {
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def json_lines(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze prospective, outcome-blind TMLR RQ4.")
    parser.add_argument("--config", default="configs/experiments/tmlr_rq4.yaml")
    parser.add_argument("--model-config", default="configs/models/llama31_ar.yaml")
    parser.add_argument("--telemetry-schema", default="configs/schemas/rq4_ar_telemetry_v1.json")
    parser.add_argument("--cache-dir", default=os.environ.get("HUGGINGFACE_HUB_CACHE"))
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--output", default="results/rq4/freeze_manifest.json")
    parser.add_argument("--prompt-ledger", default="results/rq4/prompt_render_ledger.jsonl")
    args = parser.parse_args()

    output = Path(args.output)
    prompt_ledger = Path(args.prompt_ledger)
    if output.exists() or prompt_ledger.exists():
        raise SystemExit("Refusing to overwrite immutable RQ4 freeze evidence.")
    source_commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise SystemExit("RQ4 freeze requires a clean prospective source checkout.")

    config_path = Path(args.config)
    model_config_path = Path(args.model_config)
    telemetry_schema_path = Path(args.telemetry_schema)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    model_id = str(model_config["model_id"])
    revision = str(model_config["revision"])
    if model_config["prompt_contract"]["tokenizer_revision"] != revision:
        raise SystemExit("Tokenizer revision is not identical to the model revision.")

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=args.cache_dir,
            local_files_only=True,
        )
    )
    if snapshot.name != revision:
        raise SystemExit(f"Resolved snapshot {snapshot.name!r} does not equal frozen revision.")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)

    tokenizer_files = []
    for path in sorted(snapshot.iterdir()):
        if path.name in TOKENIZER_FILENAMES and path.is_file():
            tokenizer_files.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    if not tokenizer_files:
        raise SystemExit("No tokenizer files found in the pinned snapshot.")

    chat_template = str(tokenizer.chat_template)
    sentinel_rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "TrajUQ RQ4 template freeze sentinel."}],
        tokenize=False,
        add_generation_prompt=True,
    )

    ledger_rows: list[dict] = []
    partitions: list[dict] = []
    for dataset in config["datasets"]:
        for split in ("calibration", "test"):
            contract = config["partitions"]["assets"][dataset][split]
            path = Path(args.data_root) / dataset / f"{split}.jsonl"
            rows = json_lines(path)
            observed_hash = sha256_file(path)
            if len(rows) != int(contract["rows"]) or observed_hash != str(contract["sha256"]):
                raise SystemExit(f"Frozen dataset mismatch: {dataset}/{split}")
            start = len(ledger_rows)
            for row in rows:
                prompt = str(row["prompt"])
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                ledger_rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "example_id": str(row["id"]),
                        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                        "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
                    }
                )
            partitions.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "rows": len(rows),
                    "sha256": observed_hash,
                    "ledger_start": start,
                    "ledger_count": len(rows),
                }
            )

    ledger_content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger_rows)
    atomic_text(prompt_ledger, ledger_content)
    source_files = [
        config_path,
        model_config_path,
        telemetry_schema_path,
        Path("src/trajuq/adapters/llama_ar.py"),
        Path("scripts/run_generation.py"),
    ]
    manifest = {
        "schema": "trajuq_tmlr_rq4_freeze_v1",
        "status": "PASS",
        "created_unix": int(time.time()),
        "original_authoritative_commit": config["source_commit"],
        "prospective_rq4_source_commit": source_commit,
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "snapshot_path": str(snapshot),
        "tokenizer_files": tokenizer_files,
        "chat_template_source_sha256": sha256_bytes(chat_template.encode("utf-8")),
        "sentinel_rendered_chat_template_sha256": sha256_bytes(
            sentinel_rendered.encode("utf-8")
        ),
        "telemetry_schema_sha256": sha256_file(telemetry_schema_path),
        "model_config_sha256": sha256_file(model_config_path),
        "experiment_config_sha256": sha256_file(config_path),
        "prompt_ledger": str(prompt_ledger),
        "prompt_ledger_rows": len(ledger_rows),
        "prompt_ledger_sha256": sha256_bytes(ledger_content.encode("utf-8")),
        "partitions": partitions,
        "source_files": [
            {"path": str(path).replace("\\", "/"), "sha256": sha256_file(path)}
            for path in source_files
        ],
        "outcomes_observed": False,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    atomic_text(output, manifest_text)
    atomic_text(output.with_suffix(output.suffix + ".sha256"), sha256_bytes(manifest_text.encode()) + "\n")
    print(f"PASS: RQ4 freeze -> {output}")


if __name__ == "__main__":
    main()
