#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

import yaml

from trajuq.scoring import score_record


NAMESPACE = uuid.UUID("6ddf4f6d-50dc-4a38-94d3-94ace947ff7c")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, score, and register partial RQ5 artifacts.")
    parser.add_argument("--config", default="configs/experiments/tmlr_rq5_blackbox.yaml")
    parser.add_argument("--schema", default="configs/schemas/rq5_api_provenance_v1.json")
    parser.add_argument("--freeze", default="results/rq5/TMLR_RQ5_BLACKBOX_FREEZE.json")
    parser.add_argument("--records", default="results/rq5/api/records")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--registry", default="results/rq5/validated/response_registry.jsonl")
    parser.add_argument("--manifest", default="results/rq5/validated/registry_manifest.json")
    parser.add_argument("--paper-tex", default="results/rq5/paper/generated_rq5.tex")
    parser.add_argument("--paper-manifest", default="results/rq5/paper/generated_rq5_manifest.json")
    parser.add_argument("--ambiguous-registry", default="configs/operations/rq5_ambiguous_uuid_registry.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    freeze_path = Path(args.freeze)
    freeze_sha = sha256_bytes(freeze_path.read_bytes())
    if freeze_sha != Path(str(freeze_path) + ".sha256").read_text().strip():
        raise SystemExit("RQ5 freeze hash drift")
    required = set(schema["required"])
    models = set(config["models"])
    ambiguous_contract = yaml.safe_load(Path(args.ambiguous_registry).read_text(encoding="utf-8"))
    if ambiguous_contract["freeze_sha256"] != freeze_sha:
        raise SystemExit("Ambiguous UUID registry is not bound to this freeze")
    ambiguous_by_model = defaultdict(int)
    for item in ambiguous_contract["requests"]:
        ambiguous_by_model[str(item["model"])] += 1
    expected_ids: dict[tuple[str, str], dict[str, dict]] = {}
    for dataset in config["datasets"]:
        for split in ("calibration", "test"):
            path = Path(args.data_root) / dataset / f"{split}.jsonl"
            rows = read_jsonl(path)
            expected_ids[(dataset, split)] = {str(row["id"]): row for row in rows}

    validated = []
    seen_mapping = set()
    failures = []
    for path in sorted(Path(args.records).glob("*/*/*/*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            missing = required - set(record)
            if missing:
                raise ValueError(f"missing fields {sorted(missing)}")
            if record["freeze_sha256"] != freeze_sha:
                raise ValueError("freeze hash mismatch")
            model = str(record["requested_model"])
            dataset = str(record["dataset"])
            split = str(record["split"])
            example_id = str(record["example_id"])
            repeat = int(record["repeat_index"])
            if model not in models or record["returned_model"] != model:
                raise ValueError("requested/returned model contract mismatch")
            if example_id not in expected_ids.get((dataset, split), {}):
                raise ValueError("example outside frozen partition")
            if not 0 <= repeat < int(config["decoding"]["repeated_samples"]):
                raise ValueError("repeat index outside frozen protocol")
            expected_uuid = str(uuid.uuid5(NAMESPACE, f"{freeze_sha}|{model}|{dataset}|{split}|{example_id}|{repeat}"))
            if record["request_uuid"] != expected_uuid:
                raise ValueError("request UUID mismatch")
            mapping = (model, dataset, split, example_id, repeat)
            if mapping in seen_mapping:
                raise ValueError("duplicate successful request mapping")
            seen_mapping.add(mapping)
            if any(key in record for key in ("reference", "correct", "authorization", "api_key", "secret")):
                raise ValueError("forbidden pre-outcome field")
            if int(record["http_status"]) != 200:
                raise ValueError("non-success HTTP status in successful record")
            answer = record.get("parsed_final_answer")
            source = expected_ids[(dataset, split)][example_id]
            if not isinstance(answer, str) or not answer.strip():
                correct, scorer = False, "missing_final_answer_counted_incorrect"
            else:
                scoring_text = f"FINAL: {answer}"
                correct, scorer = score_record(dataset, scoring_text, str(source["reference"]))
            usage = record.get("usage") or {}
            validated.append({
                "freeze_sha256": freeze_sha,
                "request_uuid": record["request_uuid"],
                "model": model,
                "dataset": dataset,
                "split": split,
                "example_id": example_id,
                "repeat_index": repeat,
                "correct": bool(correct),
                "scorer": scorer,
                "parsed_final_answer": answer,
                "verbalized_confidence": record.get("verbalized_confidence"),
                "finish_reason": record.get("finish_reason"),
                "latency_seconds": float(record["latency_seconds"]),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "estimated_cost_rmb": float(record.get("estimated_cost_rmb") or 0.0),
                "retry_number": int(record.get("retry_number") or 0),
                "reasoning_metadata_present": record.get("reasoning_content") is not None,
            })
        except Exception as exc:
            failures.append({"path": str(path).replace("\\", "/"), "reason": str(exc)})

    validated.sort(key=lambda row: (row["model"], row["dataset"], row["split"], row["example_id"], row["repeat_index"]))
    registry_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in validated).encode()
    atomic_bytes(Path(args.registry), registry_bytes)
    expected_total = len(config["models"]) * sum(
        int(config["partitions"]["assets"][dataset][split]["rows"])
        for dataset in config["datasets"] for split in ("calibration", "test")
    ) * int(config["decoding"]["repeated_samples"])
    manifest = {
        "schema": "trajuq_rq5_validated_registry_v1",
        "status": "PASS" if not failures else "FAIL",
        "partial": len(validated) < expected_total,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": freeze_sha,
        "validated_responses": len(validated),
        "expected_responses": expected_total,
        "validation_failures": failures,
        "registry_sha256": sha256_bytes(registry_bytes),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    atomic_bytes(Path(args.manifest), manifest_bytes)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in validated:
        grouped[(row["model"], row["dataset"])].append(row)
    labels = {
        "Pro/deepseek-ai/DeepSeek-V3.2": "DeepSeek-V3.2",
        "Pro/zai-org/GLM-5.1": "GLM-5.1",
        "Pro/moonshotai/Kimi-K2.6": "Kimi-K2.6",
    }
    table_rows = []
    operational_rows = []
    for model in config["models"]:
        for dataset in config["datasets"]:
            rows = grouped[(model, dataset)]
            target = sum(int(config["partitions"]["assets"][dataset][split]["rows"]) for split in ("calibration", "test")) * int(config["decoding"]["repeated_samples"])
            complete = len(rows) == target
            accuracy = f"{sum(int(row['correct']) for row in rows) / len(rows):.3f}" if complete and rows else "--"
            status = "complete" if complete else "pending"
            table_rows.append(f"{tex_escape(labels[model])} & {tex_escape(dataset)} & {len(rows)}/{target} & {accuracy} & {status} \\\\")
        model_rows = [row for row in validated if row["model"] == model]
        error_count = len(list((Path(args.records).parent / "errors" / model.replace("/", "__")).glob("*/*/*.json")))
        retries = sum(row["retry_number"] for row in model_rows)
        tokens = sum(row["prompt_tokens"] + row["completion_tokens"] for row in model_rows)
        cost = sum(row["estimated_cost_rmb"] for row in model_rows)
        latency = sum(row["latency_seconds"] for row in model_rows) / len(model_rows) if model_rows else 0.0
        operational_rows.append(
            f"{tex_escape(labels[model])} & {len(model_rows)} & {retries} & {ambiguous_by_model[model]} & {error_count} & {tokens} & {cost:.3f} & {latency:.2f} \\\\"
        )
    metric_rows = "\n".join(
        f"{tex_escape(labels[model])} & -- & -- & -- & pending \\\\"
        for model in config["models"]
    )
    signal_rows = "\n".join(
        f"{name} & -- & -- & pending \\\\"
        for name in (
            "Verbalized confidence",
            "Repeated-sample disagreement",
            "Answer stability / self-consistency",
            "Provider-visible metadata",
            "Pre-registered combined signal",
        )
    )
    tex = (
        "% AUTO-GENERATED from validated RQ5 artifacts; do not edit.\n"
        "\\newcommand{\\RQFiveValidatedResponses}{" + str(len(validated)) + "}\n"
        "\\newcommand{\\RQFiveExpectedResponses}{" + str(expected_total) + "}\n"
        "\\newcommand{\\RQFiveTables}{%\n"
        "\\begin{table*}[t]\\centering\\small\n"
        "\\begin{tabular}{llrrl}\\toprule\n"
        "Model & Dataset & Validated requests & Accuracy & State \\\\ \\midrule\n"
        + "\n".join(table_rows)
        + "\n\\bottomrule\\end{tabular}\n"
        "\\caption{RQ5 black-box execution and task-performance registry. Accuracy is shown only for a complete frozen model--dataset cell; incomplete cells remain explicitly pending.}\n"
        "\\label{tab:rq5-partial-registry}\\end{table*}\n"
        "\\begin{table}[t]\\centering\\small\\begin{tabular}{lrrrl}\\toprule\n"
        "Model & AUROC & AUPRC & ECE & State \\\\ \\midrule\n" + metric_rows +
        "\n\\bottomrule\\end{tabular}\\caption{RQ5 calibration and uncertainty ranking. Metrics remain pending until complete frozen cells pass validation.}\\label{tab:rq5-calibration}\\end{table}\n"
        "\\begin{table}[t]\\centering\\small\\begin{tabular}{lrrl}\\toprule\n"
        "Model & Risk@coverage & AURC & State \\\\ \\midrule\n" + metric_rows.replace(" & -- & -- & -- & pending", " & -- & -- & pending") +
        "\n\\bottomrule\\end{tabular}\\caption{RQ5 selective prediction at frozen coverage levels.}\\label{tab:rq5-selective}\\end{table}\n"
        "\\begin{table}[t]\\centering\\small\\begin{tabular}{lrrl}\\toprule\n"
        "Signal & AUROC & AURC & State \\\\ \\midrule\n" + signal_rows +
        "\n\\bottomrule\\end{tabular}\\caption{Pre-registered RQ5 black-box signal comparison.}\\label{tab:rq5-signals}\\end{table}\n"
        "\\begin{table}[t]\\centering\\small\\begin{tabular}{lrrl}\\toprule\n"
        "Observability regime & AUROC & AURC & State \\\\ \\midrule\n"
        "Diffusion trajectory & -- & -- & pending \\\\\nLlama-3.1 autoregressive & -- & -- & pending \\\\\nBlack-box API & -- & -- & pending \\\\ \\bottomrule\\end{tabular}\n"
        "\\caption{Cross-observability comparison. Values are populated only after the corresponding complete registries pass.}\\label{tab:cross-observability}\\end{table}\n"
        "\\begin{table*}[t]\\centering\\scriptsize\\begin{tabular}{lrrrrrrr}\\toprule\n"
        "Model & Valid & Retries & Ambiguous & Errors & Tokens & Cost (RMB) & Mean latency (s) \\\\ \\midrule\n"
        + "\n".join(operational_rows) +
        "\n\\bottomrule\\end{tabular}\\caption{Provisional RQ5 operational accounting from validated responses. Counts and costs update during execution and are regenerated from scratch at completion.}\\label{tab:rq5-operations}\\end{table*}}\n"
    ).encode()
    atomic_bytes(Path(args.paper_tex), tex)
    paper_manifest = {
        "schema": "trajuq_rq5_generated_paper_v1",
        "status": manifest["status"],
        "partial": manifest["partial"],
        "freeze_sha256": freeze_sha,
        "registry_sha256": manifest["registry_sha256"],
        "generated_tex_sha256": sha256_bytes(tex),
        "validated_responses": len(validated),
        "expected_responses": expected_total,
    }
    atomic_bytes(Path(args.paper_manifest), (json.dumps(paper_manifest, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({key: manifest[key] for key in ("status", "partial", "validated_responses", "expected_responses", "registry_sha256")}, indent=2))


if __name__ == "__main__":
    main()
