#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TIMING_SOURCES = {
    "llada": Path("runs/llada_gsm8k_pilot50.raw.jsonl"),
    "sdar": Path("runs/sdar_parity_trajuq.raw.jsonl"),
    "dream": Path("runs/dream_parity_trajuq.raw.jsonl"),
}
DEFAULT_PRIMARY_PLAN = Path("results/primary_matrix_plan.json")
DEFAULT_PRIMARY_CONFIG = Path("configs/experiments/primary.yaml")
DEFAULT_OUTPUT = Path("results/a100_compute_estimate.json")
DEFAULT_MARKDOWN = Path("results/a100_compute_estimate.md")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{path}:{line_number}: JSONL row is not an object"
                )
            rows.append(row)
    return rows


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def git_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
    ).strip()


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * q
    low = int(math.floor(location))
    high = int(math.ceil(location))
    if low == high:
        return ordered[low]
    weight = location - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def validate_timing_source(
    adapter: str,
    path: Path,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"{adapter}: missing timing source {path}")
    audit_path = path.with_suffix(path.suffix + ".audit.json")
    if not audit_path.exists():
        raise ValueError(
            f"{adapter}: missing timing-source raw audit {audit_path}"
        )

    audit = read_json(audit_path)
    if audit.get("protocol") != "trajuq_run_audit_v1":
        raise ValueError(
            f"{adapter}: timing-source audit protocol is not v1"
        )
    if audit.get("status") != "PASS":
        raise ValueError(f"{adapter}: timing-source audit is not PASS")
    if audit.get("stage") != "raw":
        raise ValueError(f"{adapter}: timing-source audit is not raw")
    if audit.get("git_clean") is not True:
        raise ValueError(
            f"{adapter}: timing-source audit was not recorded cleanly"
        )
    if str(audit.get("git_commit", "")) != expected_commit:
        raise ValueError(
            f"{adapter}: timing-source audit commit differs from experiment"
        )
    if str(audit.get("sha256", "")) != sha256_file(path):
        raise ValueError(
            f"{adapter}: timing-source audit SHA does not match source"
        )

    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{adapter}: timing source contains zero records")

    times: list[float] = []
    peak_memory: list[int] = []
    adapters: set[str] = set()
    for index, row in enumerate(rows):
        adapters.add(str(row.get("adapter", "")))
        metadata = row.get("generation", {}).get("metadata", {})
        value = metadata.get("wall_time_seconds")
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{adapter}: row {index} has invalid wall_time_seconds"
            ) from exc
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(
                f"{adapter}: row {index} has non-positive wall time"
            )
        times.append(seconds)

        memory_value = metadata.get("peak_gpu_memory_bytes")
        if memory_value is not None:
            try:
                memory = int(memory_value)
            except (TypeError, ValueError):
                memory = 0
            if memory > 0:
                peak_memory.append(memory)

    if adapters != {adapter}:
        raise ValueError(
            f"{adapter}: timing source adapters are {sorted(adapters)}"
        )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "records": len(rows),
        "wall_time_seconds": {
            "mean": statistics.fmean(times),
            "p50": statistics.median(times),
            "p90": quantile(times, 0.90),
            "min": min(times),
            "max": max(times),
        },
        "examples_per_hour": {
            "mean": 3600.0 / statistics.fmean(times),
            "p90_planning": 3600.0 / quantile(times, 0.90),
        },
        "peak_gpu_memory_bytes": (
            max(peak_memory) if peak_memory else None
        ),
    }


def base_budget_ratio(primary: dict[str, Any], adapter: str) -> float:
    budgets = primary["adaptive_compute"]["budgets"][adapter]
    if "base_steps" in budgets:
        return float(budgets["base_steps"]) / float(
            budgets["refined_steps"]
        )
    return float(budgets["base_denoising_steps"]) / float(
        budgets["refined_denoising_steps"]
    )


def seconds_to_hours(seconds: float) -> float:
    return seconds / 3600.0


def make_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# TrajUQ A100 campaign compute estimate",
        "",
        (
            "Planning estimate only. It is derived from frozen raw timing "
            "artifacts and exact prepared calibration/test row counts; it is "
            "not an empirical paper result."
        ),
        "",
        (
            f"Experiment commit: `{payload['experiment_git_commit']}`  "
        ),
        (
            f"Primary examples across 18 cells: "
            f"**{payload['counts']['primary_examples']}**  "
        ),
        (
            f"RQ2 auxiliary generations: "
            f"**{payload['counts']['rq2_auxiliary_generations']}**  "
        ),
        (
            f"RQ3 base-budget generations: "
            f"**{payload['counts']['rq3_base_generations']}**"
        ),
        "",
        "## Per-model projection",
        "",
        (
            "| Model | Timing records | Mean sec/example | "
            "Primary h | RQ2 h | RQ3 base h | Total h |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["per_model"]:
        lines.append(
            "| {adapter} | {records} | {mean:.3f} | {primary:.2f} | "
            "{rq2:.2f} | {rq3:.2f} | {total:.2f} |".format(
                adapter=row["adapter"],
                records=row["timing"]["records"],
                mean=row["timing"]["wall_time_seconds"]["mean"],
                primary=row["mean_projection_hours"]["primary"],
                rq2=row["mean_projection_hours"]["rq2_repeated"],
                rq3=row["mean_projection_hours"]["rq3_base"],
                total=row["mean_projection_hours"]["total"],
            )
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            (
                "- Mean-timing projection: "
                f"**{payload['aggregate']['mean_projection_hours']:.2f} "
                "A100-hours**."
            ),
            (
                "- P90-per-example planning projection: "
                f"**{payload['aggregate']['p90_planning_hours']:.2f} "
                "A100-hours**."
            ),
            (
                "- These are serial single-A100 hours and exclude model "
                "download/load startup, CPU scoring/analysis, retries, and "
                "optional RQ4."
            ),
            (
                "- The P90 projection is a conservative planning sum, not a "
                "statistical 90% confidence bound on campaign duration."
            ),
            (
                "- SDAR/Dream estimates initially use their 20-prompt parity "
                "telemetry; LLaDA uses the 50-example GSM8K pilot. Prompt "
                "length and dataset mix can shift realized throughput."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate serial A100-hours for the frozen TrajUQ primary, RQ2, "
            "and RQ3 campaigns from audited on-GPU timing traces."
        )
    )
    parser.add_argument(
        "--primary-plan",
        default=str(DEFAULT_PRIMARY_PLAN),
    )
    parser.add_argument(
        "--primary-config",
        default=str(DEFAULT_PRIMARY_CONFIG),
    )
    parser.add_argument(
        "--timing-source",
        action="append",
        default=[],
        metavar="ADAPTER=PATH",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN))
    args = parser.parse_args()

    plan_path = Path(args.primary_plan)
    config_path = Path(args.primary_config)
    if not plan_path.exists():
        raise SystemExit(
            "Missing primary matrix plan. Run primary readiness and "
            "scripts/plan_primary_matrix.py first."
        )
    if not config_path.exists():
        raise SystemExit(f"Missing primary config: {config_path}")

    plan = read_json(plan_path)
    if plan.get("protocol") != "trajuq_primary_matrix_plan_v2":
        raise SystemExit("Primary matrix plan protocol is not v2.")
    if plan.get("status") != "PASS":
        raise SystemExit("Primary matrix plan is not PASS.")

    expected_commit = str(plan.get("experiment_git_commit", ""))
    if not expected_commit:
        raise SystemExit("Primary plan has no experiment_git_commit.")
    if git_head() != expected_commit:
        raise SystemExit(
            "Current HEAD differs from the frozen experiment commit."
        )
    if not git_clean():
        raise SystemExit(
            "Compute estimation requires the clean frozen experiment checkout."
        )

    primary = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if sha256_file(config_path) != str(
        plan.get("primary_config_sha256", "")
    ):
        raise SystemExit("Primary config hash differs from primary plan.")

    timing_sources = dict(DEFAULT_TIMING_SOURCES)
    for item in args.timing_source:
        if "=" not in item:
            raise SystemExit("--timing-source must use ADAPTER=PATH")
        adapter, path = item.split("=", 1)
        timing_sources[adapter.strip()] = Path(path.strip())

    cells = list(plan.get("cells", []))
    expected_cells = int(plan.get("expected_cells", 0))
    if len(cells) != expected_cells or expected_cells != 18:
        raise SystemExit(
            f"Expected complete 18-cell primary plan; got {len(cells)}."
        )

    per_model: list[dict[str, Any]] = []
    total_primary_examples = 0
    total_rq2_generations = 0
    total_rq3_generations = 0
    aggregate_mean_seconds = 0.0
    aggregate_p90_seconds = 0.0

    repeated = primary["baselines"]["expensive"]["repeated_sampling"]
    samples = int(repeated["samples"])

    for adapter in ("llada", "sdar", "dream"):
        adapter_cells = [
            cell for cell in cells if str(cell.get("adapter")) == adapter
        ]
        if len(adapter_cells) != 6:
            raise SystemExit(
                f"{adapter}: expected 6 primary cells; got "
                f"{len(adapter_cells)}"
            )

        timing = validate_timing_source(
            adapter,
            timing_sources[adapter],
            expected_commit=expected_commit,
        )
        mean_seconds = float(timing["wall_time_seconds"]["mean"])
        p90_seconds = float(timing["wall_time_seconds"]["p90"])

        example_count = 0
        dataset_counts: dict[str, dict[str, int]] = {}
        for cell in adapter_cells:
            dataset = str(cell["dataset"])
            cal_path = Path(str(cell["calibration"]["input"]))
            test_path = Path(str(cell["test"]["input"]))
            cal_count = count_jsonl(cal_path)
            test_count = count_jsonl(test_path)
            if cal_count <= 0 or test_count <= 0:
                raise SystemExit(
                    f"{adapter}:{dataset}: empty prepared partition"
                )
            dataset_counts[dataset] = {
                "calibration": cal_count,
                "test": test_count,
                "total": cal_count + test_count,
            }
            example_count += cal_count + test_count

        ratio = base_budget_ratio(primary, adapter)
        primary_generations = example_count
        rq2_generations = example_count * samples
        rq3_generations = example_count

        mean_primary = primary_generations * mean_seconds
        mean_rq2 = rq2_generations * mean_seconds
        mean_rq3 = rq3_generations * mean_seconds * ratio
        p90_primary = primary_generations * p90_seconds
        p90_rq2 = rq2_generations * p90_seconds
        p90_rq3 = rq3_generations * p90_seconds * ratio

        total_primary_examples += primary_generations
        total_rq2_generations += rq2_generations
        total_rq3_generations += rq3_generations
        aggregate_mean_seconds += mean_primary + mean_rq2 + mean_rq3
        aggregate_p90_seconds += p90_primary + p90_rq2 + p90_rq3

        per_model.append(
            {
                "adapter": adapter,
                "timing": timing,
                "dataset_counts": dataset_counts,
                "prepared_examples": example_count,
                "repeated_samples": samples,
                "rq3_base_to_refined_budget_ratio": ratio,
                "mean_projection_hours": {
                    "primary": seconds_to_hours(mean_primary),
                    "rq2_repeated": seconds_to_hours(mean_rq2),
                    "rq3_base": seconds_to_hours(mean_rq3),
                    "total": seconds_to_hours(
                        mean_primary + mean_rq2 + mean_rq3
                    ),
                },
                "p90_planning_hours": {
                    "primary": seconds_to_hours(p90_primary),
                    "rq2_repeated": seconds_to_hours(p90_rq2),
                    "rq3_base": seconds_to_hours(p90_rq3),
                    "total": seconds_to_hours(
                        p90_primary + p90_rq2 + p90_rq3
                    ),
                },
            }
        )

    payload = {
        "protocol": "trajuq_a100_compute_estimate_v1",
        "status": "PASS",
        "scope": [
            "primary_generation",
            "rq2_five_sample_auxiliary_generation",
            "rq3_base_budget_generation",
        ],
        "excludes": [
            "model_download_and_load_startup",
            "cpu_scoring_and_analysis",
            "failed_or_retried_runs",
            "optional_rq4",
        ],
        "planning_only": True,
        "experiment_git_commit": expected_commit,
        "primary_plan": str(plan_path),
        "primary_plan_sha256": sha256_file(plan_path),
        "primary_config": str(config_path),
        "primary_config_sha256": sha256_file(config_path),
        "counts": {
            "primary_examples": total_primary_examples,
            "rq2_auxiliary_generations": total_rq2_generations,
            "rq3_base_generations": total_rq3_generations,
        },
        "per_model": per_model,
        "aggregate": {
            "mean_projection_hours": seconds_to_hours(
                aggregate_mean_seconds
            ),
            "p90_planning_hours": seconds_to_hours(
                aggregate_p90_seconds
            ),
        },
        "caveats": [
            (
                "Timing traces are small planning samples rather than the "
                "full benchmark prompt-length distribution."
            ),
            (
                "RQ3 base-runtime is estimated by linear scaling with the "
                "frozen base/refined denoising-step ratio."
            ),
            (
                "The p90 planning sum is not a statistical confidence bound "
                "for end-to-end campaign duration."
            ),
        ],
    }

    output = Path(args.output)
    markdown = Path(args.markdown)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown.write_text(make_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
