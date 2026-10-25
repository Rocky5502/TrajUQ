#!/usr/bin/env python
"""Build the artifact-derived compute ledger, summary, LaTeX, and hash manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "compute_budget"


def load(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def tex_escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(a, b)
    return text


def params(value, active=None) -> str:
    if value is None:
        return "not disclosed"
    def compact(n):
        return f"{n / 1_000_000_000:.3f}B" if n >= 1_000_000_000 else f"{n / 1_000_000:.1f}M"
    result = compact(value)
    if active is not None and active != value:
        result += f" total / {compact(active)} active"
    return result


def iso_from_epoch(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def classify_stage(value: object) -> str:
    stage = str(value or "unknown").lower()
    if stage == "primary":
        return "primary"
    if stage in {"rq2", "repeated", "repeated_sampling"}:
        return "rq2"
    if stage in {"rq3", "adaptive", "adaptive_compute"}:
        return "rq3"
    if stage in {"rq4", "llama31"}:
        return "rq4"
    return stage


def audited_final_run(run: dict, expected_commit: str | None) -> bool:
    return bool(
        run.get("audit_present")
        and run.get("audit_status") == "PASS"
        and run.get("audit_git_clean") is True
        and run.get("audit_git_commit") == expected_commit
        and run.get("audit_stage") == "raw"
        and run.get("audit_binds_output") is True
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = load(ROOT / "configs" / "operations" / "model_parameter_registry.json")
    env = load(OUT / "current_environment.json", {})
    history = load(OUT / "historical_evidence.json", {})
    rq5_cost = load(ROOT / "results" / "rq5" / "api" / "cost_ledger.json", {})
    rq5_manifest = load(ROOT / "results" / "rq5" / "validated" / "registry_manifest.json", {})
    ambiguous_path = ROOT / "configs" / "operations" / "rq5_ambiguous_uuid_registry.yaml"
    ambiguous_count = ambiguous_path.read_text(encoding="utf-8").count("request_uuid:") if ambiguous_path.exists() else 0

    ledger = []
    for run in env.get("raw_runs", []):
        if "read_error" in run:
            continue
        seconds = float(run.get("generation_seconds") or 0.0)
        end = datetime.fromisoformat(run["file_end_utc"])
        start = end.timestamp() - seconds
        stage = classify_stage(run.get("stage"))
        finalized = audited_final_run(run, (env.get("git") or {}).get("commit"))
        authoritative = stage in {"primary", "rq2", "rq3", "rq4"}
        ledger.append({
            "record_type": "authoritative_experimental_compute" if authoritative else "development_diagnostic_recovery_compute",
            "experiment_stage": stage,
            "model": run.get("model"),
            "dataset": run.get("dataset"),
            "split": run.get("split"),
            "start_utc": iso_from_epoch(start),
            "end_utc": run.get("file_end_utc"),
            "wall_clock_hours": seconds / 3600.0,
            "gpu_count": int((env.get("gpu") or {}).get("count") or 1),
            "gpu_model": (env.get("gpu") or {}).get("name"),
            "gpu_memory_mib": (env.get("gpu") or {}).get("memory_mib"),
            "gpu_hours": seconds / 3600.0 * int((env.get("gpu") or {}).get("count") or 1),
            "host": env.get("host"),
            "repository_commit": (env.get("git") or {}).get("commit"),
            "runtime_environment_sha256": None,
            "output_path": run.get("path"),
            "output_sha256": run.get("output_sha256"),
            "audit_sha256": run.get("audit_sha256"),
            "examples": run.get("rows"),
            "peak_gpu_memory_bytes": run.get("peak_gpu_memory_bytes"),
            "contributed_to_final_results": finalized and authoritative,
            "measurement_method": "sum of artifact-recorded per-example generation.metadata.wall_time_seconds; start is derived from output mtime minus that sum",
            "measurement_limit": "excludes model loading, scoring, and inter-example orchestration overhead",
        })

    previous = (history.get("previous_rental") or {})
    ledger.append({
        "record_type": "development_diagnostic_recovery_compute",
        "experiment_stage": "previous_rental_envelope",
        "model": None,
        "dataset": None,
        "start_utc": previous.get("started_at_utc"),
        "end_utc": previous.get("deadline_at_utc"),
        "wall_clock_hours": previous.get("rental_envelope_hours"),
        "gpu_count": previous.get("gpu_count"),
        "gpu_model": previous.get("gpu_model"),
        "gpu_hours": previous.get("active_gpu_hours"),
        "repository_commit": previous.get("commit_at_clock_capture"),
        "contributed_to_final_results": False,
        "measurement_method": "preserved rental-clock artifact",
        "measurement_limit": previous.get("limitation"),
    })

    # Bind the environment before writing ledger rows that refer to it.
    environment_path = OUT / "environment_manifest.json"
    atomic(environment_path, json_bytes(env))
    environment_sha = sha(environment_path)
    for row in ledger:
        if row["record_type"] == "authoritative_experimental_compute":
            row["runtime_environment_sha256"] = environment_sha

    ledger_path = OUT / "compute_ledger.jsonl"
    ledger_data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger).encode()
    atomic(ledger_path, ledger_data)

    counted = [r for r in ledger if r["record_type"] == "authoritative_experimental_compute" and r["contributed_to_final_results"]]
    stage_hours = {name: sum(float(r.get("gpu_hours") or 0) for r in counted if r["experiment_stage"] == name) for name in ("primary", "rq2", "rq3", "rq4")}

    rq5_registry = ROOT / "results" / "rq5" / "validated" / "response_registry.jsonl"
    rq5_rows = []
    if rq5_registry.exists():
        with rq5_registry.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rq5_rows.append(json.loads(line))
    input_tokens = sum(int(r.get("prompt_tokens") or 0) for r in rq5_rows)
    output_tokens = sum(int(r.get("completion_tokens") or 0) for r in rq5_rows)
    reasoning_tokens = sum(int(r.get("reasoning_tokens") or 0) for r in rq5_rows)
    latencies = [float(r["latency_seconds"]) for r in rq5_rows if isinstance(r.get("latency_seconds"), (int, float))]
    retries = sum(int(r.get("retry_number") or 0) for r in rq5_rows)
    start_current = (history.get("current_rental") or {}).get("started_at_utc")
    elapsed = None
    if start_current and env.get("captured_utc"):
        elapsed = (datetime.fromisoformat(env["captured_utc"]) - datetime.fromisoformat(start_current)).total_seconds() / 3600

    measured_diagnostic = sum(float(r.get("gpu_hours") or 0) for r in ledger if r["record_type"] == "development_diagnostic_recovery_compute" and r.get("gpu_hours") is not None)
    summary = {
        "schema": "trajuq_compute_summary_v1",
        "status": "PROVISIONAL" if any(not r["contributed_to_final_results"] for r in ledger if r["record_type"] == "authoritative_experimental_compute") or rq5_manifest.get("partial", True) else "FINAL",
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_gpu_hours": {
            "primary": stage_hours["primary"],
            "rq2": stage_hours["rq2"],
            "rq3": stage_hours["rq3"],
            "rq4": stage_hours["rq4"],
            "total": sum(stage_hours.values()),
            "definition": "sum of finalized/audited per-example generation wall times times GPU count; utilization is not applied",
        },
        "diagnostic_gpu_hours": {
            "known_measured": measured_diagnostic,
            "previous_rental_active": None,
            "limitation": previous.get("limitation"),
        },
        "project_gpu_hours": None,
        "current_rental": {"started_at_utc": start_current, "elapsed_wall_hours": elapsed},
        "rq5": {
            "provider": "SiliconFlow",
            "requests_saved": rq5_cost.get("completed_requests", 0),
            "valid_responses": len(rq5_rows),
            "validation_failures": len(rq5_manifest.get("validation_failures") or []),
            "ambiguous_responses": ambiguous_count,
            "retries": retries,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens_explicitly_reported": reasoning_tokens,
            "cost_rmb": rq5_cost.get("cumulative_cost_rmb", 0.0),
            "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
            "median_latency_seconds": statistics.median(latencies) if latencies else None,
            "provider_hardware": None,
            "provider_gpu_hours": None,
            "limitation": "Provider-side GPU type, GPU count, and GPU-hours are not observable through the black-box API.",
        },
        "model_parameter_registry_sha256": sha(ROOT / "configs" / "operations" / "model_parameter_registry.json"),
        "environment_manifest_sha256": environment_sha,
        "compute_ledger_sha256": sha(ledger_path),
    }
    summary_path = OUT / "compute_summary.json"
    atomic(summary_path, json_bytes(summary))

    models = {m["key"]: m for m in registry["models"]}
    table_labels = {
        "dream": "Dream-7B", "llada": "LLaDA-MoE-7B-A1B", "sdar": "SDAR-1.7B",
        "llama31": "Llama-3.1-8B", "deepseek_v32": "DeepSeek-V3.2",
        "glm51": "GLM-5.1", "kimi_k26": "Kimi-K2.6",
    }
    def row(key, stage, access, usage):
        m=models[key]
        p=tex_escape(params(m["total_parameters"], m["active_parameters"]))
        precision = "provider-managed" if m["total_parameters"] is None else tex_escape(m["precision"])
        return f"{tex_escape(table_labels[key])} ({stage}) & {p} & {access} & {precision} & {usage} \\\\"

    primary_hours = stage_hours["primary"]
    r23 = stage_hours["rq2"] + stage_hours["rq3"]
    rq4 = stage_hours["rq4"]
    total = sum(stage_hours.values())
    cost = float(summary["rq5"]["cost_rmb"] or 0)
    req = int(summary["rq5"]["requests_saved"])
    tex = [
        "% AUTO-GENERATED from compute artifacts; do not edit.",
        f"\\newcommand{{\\ComputeLedgerStatus}}{{{summary['status'].lower()}}}",
        f"\\newcommand{{\\TotalPrimaryGPUHours}}{{{primary_hours:.2f}}}",
        f"\\newcommand{{\\TotalRQTwoThreeGPUHours}}{{{r23:.2f}}}",
        f"\\newcommand{{\\TotalRQFourGPUHours}}{{{rq4:.2f}}}",
        f"\\newcommand{{\\TotalAuthoritativeGPUHours}}{{{total:.2f}}}",
        f"\\newcommand{{\\TotalDiagnosticGPUHours}}{{{measured_diagnostic:.2f} measured; prior rental unavailable}}",
        "\\newcommand{\\TotalProjectGPUHours}{not exactly reconstructable}",
        f"\\newcommand{{\\RQFiveRequests}}{{{req:,}}}",
        f"\\newcommand{{\\RQFiveInputTokens}}{{{input_tokens:,}}}",
        f"\\newcommand{{\\RQFiveOutputTokens}}{{{output_tokens:,}}}",
        f"\\newcommand{{\\RQFiveCostRMB}}{{{cost:.2f}}}",
        f"\\newcommand{{\\CurrentRentalElapsedHours}}{{{elapsed:.1f}}}" if elapsed is not None else "\\newcommand{\\CurrentRentalElapsedHours}{unavailable}",
        f"\\newcommand{{\\ComputeGPUName}}{{{tex_escape(str((env.get('gpu') or {}).get('name') or 'unavailable'))}}}",
        f"\\newcommand{{\\ComputeGPUMemoryMiB}}{{{int((env.get('gpu') or {}).get('memory_mib') or 0):,}}}",
        f"\\newcommand{{\\ComputeDriverVersion}}{{{tex_escape(str((env.get('gpu') or {}).get('driver') or 'unavailable'))}}}",
        f"\\newcommand{{\\ComputeCUDARuntime}}{{{tex_escape(str((env.get('runtime') or {}).get('cuda_runtime') or 'unavailable'))}}}",
        f"\\newcommand{{\\ComputeTorchVersion}}{{{tex_escape(str((env.get('runtime') or {}).get('torch') or 'unavailable'))}}}",
        f"\\newcommand{{\\ComputeTransformersVersion}}{{{tex_escape(str((env.get('runtime') or {}).get('transformers') or 'unavailable'))}}}",
        f"\\newcommand{{\\ComputeFlashAttentionVersion}}{{{tex_escape(str((env.get('runtime') or {}).get('flash_attention') or 'unavailable'))}}}",
        f"\\newcommand{{\\ComputeTritonVersion}}{{{tex_escape(str((env.get('runtime') or {}).get('triton') or 'unavailable'))}}}",
        f"\\newcommand{{\\ComputeCPUModel}}{{{tex_escape(str((env.get('cpu') or {}).get('model') or 'unavailable'))}}}",
        f"\\newcommand{{\\ComputeLogicalCPUs}}{{{int((env.get('cpu') or {}).get('logical_cpus') or 0)}}}",
        f"\\newcommand{{\\ComputeRAMGiB}}{{{float((env.get('memory') or {}).get('ram_bytes') or 0) / 2**30:.1f}}}",
        f"\\newcommand{{\\ComputeDataDiskGiB}}{{{float((env.get('storage') or {}).get('total_bytes') or 0) / 2**30:.1f}}}",
        f"\\newcommand{{\\ComputeOS}}{{{tex_escape(str((env.get('os') or {}).get('description') or 'unavailable'))}}}",
        f"\\newcommand{{\\ComputePythonVersion}}{{{tex_escape(str((env.get('os') or {}).get('python') or 'unavailable'))}}}",
        "\\newcommand{\\ComputeBudgetTable}{%",
        "\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.20\\textwidth}>{\\raggedright\\arraybackslash}p{0.19\\textwidth}>{\\raggedright\\arraybackslash}p{0.16\\textwidth}>{\\raggedright\\arraybackslash}p{0.15\\textwidth}>{\\raggedright\\arraybackslash}p{0.22\\textwidth}@{}}\\toprule",
        "Model/stage & Parameters & Hardware/access & Precision & Measured usage \\\\ \\midrule",
        row("dream", "primary", "A100 80GB", "\\TotalPrimaryGPUHours{} shared primary"),
        row("llada", "primary", "A100 80GB", "\\TotalPrimaryGPUHours{} shared primary"),
        row("sdar", "primary", "A100 80GB", "\\TotalPrimaryGPUHours{} shared primary"),
        row("llama31", "RQ4", "A100 80GB", "\\TotalRQFourGPUHours{}"),
        row("deepseek_v32", "RQ5", "SiliconFlow API", "\\RQFiveRequests{} shared API requests"),
        row("glm51", "RQ5", "SiliconFlow API", "\\RQFiveRequests{} shared API requests"),
        row("kimi_k26", "RQ5", "SiliconFlow API", "\\RQFiveRequests{} shared API requests"),
        "\\bottomrule\\end{tabular}",
        "\\caption{Artifact-derived compute snapshot. Shared totals are not repeated per model when the registry currently aggregates a stage. Parameter counts are exact pinned-artifact tensor counts for open models; hosted-model counts are not disclosed. Final release regenerates this table from the immutable completed ledger.}",
        "\\label{tab:compute-budget}\\end{table*}}",
        "\\newcommand{\\ComputeBudgetAppendixTable}{%",
        "\\begingroup\\footnotesize\\setlength{\\LTpre}{0pt}\\setlength{\\LTpost}{0pt}",
        "\\begin{longtable}{@{}p{0.08\\textwidth}p{0.20\\textwidth}p{0.13\\textwidth}p{0.10\\textwidth}rrp{0.15\\textwidth}@{}}",
        "\\caption{Detailed finalized-run accounting. Active and unaudited files are retained in the machine-readable ledger but excluded here until their audit passes. Per-example generation times exclude model-load and orchestration overhead.}\\label{tab:compute-budget-detail}\\\\",
        "\\toprule Stage & Model & Dataset & Split & Examples & GPU-h & Evidence \\\\ \\midrule\\endfirsthead",
        "\\toprule Stage & Model & Dataset & Split & Examples & GPU-h & Evidence \\\\ \\midrule\\endhead",
        "\\midrule\\multicolumn{7}{r}{Continued on next page}\\\\\\endfoot",
        "\\bottomrule\\endlastfoot",
    ]
    for r in counted:
        tex.append(f"{tex_escape(str(r['experiment_stage']))} & {tex_escape(str(r.get('model') or '---'))} & {tex_escape(str(r.get('dataset') or '---'))} & {tex_escape(str(r.get('split') or '---'))} & {int(r.get('examples') or 0)} & {float(r.get('gpu_hours') or 0):.3f} & SHA-bound + audit \\\\ ")
    if not counted:
        tex.append("--- & --- & --- & --- & 0 & 0.000 & pending \\\\ ")
    tex += [
        "\\end{longtable}\\endgroup}",
    ]
    tex_path = OUT / "generated_compute.tex"
    atomic(tex_path, ("\n".join(tex) + "\n").encode())

    files = [environment_path, ledger_path, summary_path, tex_path, ROOT / "configs" / "operations" / "model_parameter_registry.json", OUT / "historical_evidence.json"]
    manifest = {
        "schema": "trajuq_compute_budget_manifest_v1",
        "status": "PASS",
        "summary_status": summary["status"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "files": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files],
        "audit_notes": [
            "No utilization percentage is multiplied into GPU-hours.",
            "Previous-rental active GPU time is unavailable and excluded from totals.",
            "Black-box provider hardware and GPU-hours are not observable and are not reported.",
            "Active unaudited run files are excluded from authoritative totals until audit PASS."
        ],
    }
    atomic(OUT / "compute_budget_manifest.json", json_bytes(manifest))
    print(json.dumps({"status":"PASS","summary_status":summary["status"],"authoritative_gpu_hours":total,"rq5_requests":req,"rq5_cost_rmb":cost}, indent=2))


if __name__ == "__main__":
    main()
