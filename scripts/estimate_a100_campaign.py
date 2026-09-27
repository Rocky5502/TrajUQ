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
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
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
    model_ids: set[str] = set()
    source_commits: set[str] = set()
    source_clean_values: set[bool] = set()
    model_revisions: set[str] = set()
    for index, row in enumerate(rows):
        adapters.add(str(row.get("adapter", "")))
        model_ids.add(str(row.get("model", "")).strip())
        record_metadata = row.get("metadata", {})
        if not isinstance(record_metadata, dict):
            raise ValueError(
                f"{adapter}: row {index} has invalid record metadata"
            )
        source_commit = str(record_metadata.get("source_git_commit", "")).strip()
        source_clean = record_metadata.get("source_git_clean")
        if not source_commit:
            raise ValueError(
                f"{adapter}: row {index} has no source_git_commit"
            )
        if source_clean is not True:
            raise ValueError(
                f"{adapter}: row {index} was not generated from a clean checkout"
            )
        source_commits.add(source_commit)
        source_clean_values.add(True)

        metadata = row.get("generation", {}).get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(
                f"{adapter}: row {index} has invalid generation metadata"
            )
        revision = str(metadata.get("model_revision", "")).strip()
        if not revision:
            raise ValueError(
                f"{adapter}: row {index} has no model_revision"
            )
        model_revisions.add(revision)
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
    if "" in model_ids or len(model_ids) != 1:
        raise ValueError(
            f"{adapter}: timing source model IDs are not uniform: "
            f"{sorted(model_ids)}"
        )
    observed_model_id = next(iter(model_ids))
    if expected_model_id is not None and observed_model_id != expected_model_id:
        raise ValueError(
            f"{adapter}: timing source model ID {observed_model_id!r} differs "
            f"from frozen primary model {expected_model_id!r}"
        )
    if source_commits != {expected_commit}:
        raise ValueError(
            f"{adapter}: timing source commit(s) {sorted(source_commits)} "
            f"differ from experiment commit {expected_commit}"
        )
    if source_clean_values != {True}:
        raise ValueError(
            f"{adapter}: timing source does not have uniform clean provenance"
        )
    audit_source_commits = {
        str(value)
        for value in audit.get("source_git_commits", [])
    }
    if audit_source_commits != {expected_commit}:
        raise ValueError(
            f"{adapter}: timing-source audit row provenance is not bound "
            "to the experiment commit"
        )
    if audit.get("source_git_clean_values") != [True]:
        raise ValueError(
            f"{adapter}: timing-source audit does not attest clean source rows"
        )
    if len(model_revisions) != 1:
        raise ValueError(
            f"{adapter}: timing source model revisions are not uniform: "
            f"{sorted(model_revisions)}"
        )
    observed_revision = next(iter(model_revisions))
    if (
        expected_model_revision is not None
        and observed_revision != expected_model_revision
    ):
        raise ValueError(
            f"{adapter}: timing source model revision {observed_revision!r} "
            f"differs from frozen primary revision {expected_model_revision!r}"
        )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "records": len(rows),
        "source_git_commit": next(iter(source_commits)),
        "source_git_clean": True,
        "model_id": observed_model_id,
        "model_revision": observed_revision,
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


def repeated_prepared_example_count(
    dataset_counts: dict[str, dict[str, int]],
    eligible_datasets: list[str],
) -> int:
    missing = [
        name for name in eligible_datasets
        if name not in dataset_counts
    ]
    if missing:
        raise ValueError(
            "Repeated-sampling eligible datasets are missing from prepared "
            f"counts: {missing}"
        )
    return sum(
        int(dataset_counts[name]["total"])
        for name in eligible_datasets
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
            f"**{payload['counts']['rq3_base_generations']}**  "
        ),
        (
            f"Total generation calls in the estimated scope: "
            f"**{payload['counts']['total_generation_calls']}**"
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
                "- Mean-timing 24/7 single-A100 duration: "
                f"**{payload['aggregate']['mean_projection_days_24h']:.2f} days**."
            ),
            (
                "- P90-planning 24/7 single-A100 duration: "
                f"**{payload['aggregate']['p90_planning_days_24h']:.2f} days**."
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
    seeds = [int(seed) for seed in repeated["seeds"]]
    repeated_eligible = [
        str(name) for name in repeated.get("eligible_datasets", [])
    ]
    primary_datasets = [str(name) for name in primary.get("datasets", [])]
    if samples <= 0 or len(seeds) != samples or len(set(seeds)) != samples:
        raise SystemExit(
            "Repeated-sampling config must have one unique seed per sample."
        )
    if (
        not repeated_eligible
        or len(repeated_eligible) != len(set(repeated_eligible))
        or not set(repeated_eligible).issubset(set(primary_datasets))
    ):
        raise SystemExit(
            "Repeated-sampling eligible datasets are empty, duplicated, or "
            "outside the frozen primary dataset set."
        )

    for adapter in ("llada", "sdar", "dream"):
        adapter_cells = [
            cell for cell in cells if str(cell.get("adapter")) == adapter
        ]
        if len(adapter_cells) != 6:
            raise SystemExit(
                f"{adapter}: expected 6 primary cells; got "
                f"{len(adapter_cells)}"
            )

        frozen_model_ids = {
            str(cell.get("model_id", ""))
            for cell in adapter_cells
        }
        frozen_model_revisions = {
            str(cell.get("model_revision", ""))
            for cell in adapter_cells
        }
        if len(frozen_model_ids) != 1 or "" in frozen_model_ids:
            raise SystemExit(
                f"{adapter}: primary plan model IDs are not uniform"
            )
        if len(frozen_model_revisions) != 1 or "" in frozen_model_revisions:
            raise SystemExit(
                f"{adapter}: primary plan model revisions are not uniform"
            )

        timing = validate_timing_source(
            adapter,
            timing_sources[adapter],
            expected_commit=expected_commit,
            expected_model_id=next(iter(frozen_model_ids)),
            expected_model_revision=next(iter(frozen_model_revisions)),
        )
        mean_seconds = float(timing["wall_time_seconds"]["mean"])
        p90_seconds = float(timing["wall_time_seconds"]["p90"])

        example_count = 0
        dataset_counts: dict[str, dict[str, int]] = {}
        for cell in adapter_cells:
            dataset = str(cell["dataset"])
            cal_path = Path(str(cell["calibration"]["input"]))
            test_path = Path(str(cell["test"]["input"]))
            if not cal_path.exists() or not test_path.exists():
                raise SystemExit(
                    f"{adapter}:{dataset}: prepared partition is missing"
                )
            if sha256_file(cal_path) != str(
                cell["calibration"].get("input_sha256", "")
            ):
                raise SystemExit(
                    f"{adapter}:{dataset}: calibration input drifted"
                )
            if sha256_file(test_path) != str(
                cell["test"].get("input_sha256", "")
            ):
                raise SystemExit(
                    f"{adapter}:{dataset}: test input drifted"
                )
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
        rq2_example_count = repeated_prepared_example_count(
            dataset_counts,
            repeated_eligible,
        )
        primary_generations = example_count
        rq2_generations = rq2_example_count * samples
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
                "repeated_eligible_datasets": repeated_eligible,
                "repeated_prepared_examples": rq2_example_count,
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
        "repeated_eligible_datasets": repeated_eligible,
        "repeated_expected_cells": 3 * len(repeated_eligible),
        "experiment_git_commit": expected_commit,
        "primary_plan": str(plan_path),
        "primary_plan_sha256": sha256_file(plan_path),
        "primary_config": str(config_path),
        "primary_config_sha256": sha256_file(config_path),
        "counts": {
            "primary_examples": total_primary_examples,
            "rq2_auxiliary_generations": total_rq2_generations,
            "rq3_base_generations": total_rq3_generations,
            "total_generation_calls": (
                total_primary_examples
                + total_rq2_generations
                + total_rq3_generations
            ),
        },
        "per_model": per_model,
        "aggregate": {
            "mean_projection_hours": seconds_to_hours(
                aggregate_mean_seconds
            ),
            "p90_planning_hours": seconds_to_hours(
                aggregate_p90_seconds
            ),
            "mean_projection_days_24h": seconds_to_hours(
                aggregate_mean_seconds
            ) / 24.0,
            "p90_planning_days_24h": seconds_to_hours(
                aggregate_p90_seconds
            ) / 24.0,
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
