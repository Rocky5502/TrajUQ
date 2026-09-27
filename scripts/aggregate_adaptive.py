#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from trajuq.adaptive_aggregation import AdaptiveCell, hierarchical_macro_adaptive
from trajuq.io import read_jsonl
from trajuq.provenance import (
    analysis_provenance,
    binding_matches,
    compact_binding_matches,
    sha256_file,
)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.execute_adaptive_plan import validate_plan_provenance  # noqa: E402


def load_prediction_cell(
    spec: dict,
    fraction_key: str,
    expected_commit: str | None = None,
) -> AdaptiveCell:
    label = str(spec["cell"])
    evaluation = spec.get("evaluations", {}).get(fraction_key)
    if evaluation is None:
        raise SystemExit(f"{label}: no adaptive evaluation for {fraction_key}")

    prediction_path = Path(evaluation["predictions"])
    result_path = Path(evaluation["result"])
    if not prediction_path.exists():
        raise SystemExit(f"{label}: missing adaptive predictions {prediction_path}")
    if not result_path.exists():
        raise SystemExit(f"{label}: missing adaptive result {result_path}")

    rows = list(read_jsonl(prediction_path))
    if not rows:
        raise SystemExit(f"{label}: empty adaptive predictions")
    ids = [str(row["example_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"{label}: duplicate adaptive prediction IDs")

    expected_rows = list(
        read_jsonl(spec["provenance"]["test"]["refined_scored"])
    )
    expected_ids = [str(row["example_id"]) for row in expected_rows]
    if (
        not expected_ids
        or len(expected_ids) != len(set(expected_ids))
        or set(ids) != set(expected_ids)
    ):
        raise SystemExit(
            f"{label}: adaptive predictions do not exactly cover "
            "the frozen refined test target"
        )

    result = json.loads(result_path.read_text(encoding="utf-8"))

    if expected_commit is not None:
        source_paths = {
            "cal_base": str(spec["calibration_base"]["scored_output"]),
            "cal_refined": str(
                spec["provenance"]["calibration"]["refined_scored"]
            ),
            "test_base": str(spec["test_base"]["scored_output"]),
            "test_refined": str(
                spec["provenance"]["test"]["refined_scored"]
            ),
        }
        expected_sources: dict[str, tuple[str, str]] = {}
        expected_sha256: dict[str, str] = {}
        for name, raw_path in source_paths.items():
            path = Path(raw_path)
            if not path.is_file():
                raise SystemExit(
                    f"{label}: adaptive evaluation source missing: {path}"
                )
            digest = sha256_file(path)
            expected_sources[name] = (str(path), digest)
            expected_sha256[name] = digest
        if not binding_matches(
            result.get("provenance", {}),
            expected_commit=expected_commit,
            expected_sources=expected_sources,
        ):
            raise SystemExit(
                f"{label}: adaptive result source binding is stale"
            )
        if any(
            not compact_binding_matches(
                row.get("analysis_provenance", {}),
                expected_commit=expected_commit,
                expected_sha256=expected_sha256,
            )
            for row in rows
        ):
            raise SystemExit(
                f"{label}: adaptive prediction source binding is stale"
            )

    if result.get("protocol") != "trajuq_adaptive_evaluation_v2":
        raise SystemExit(f"{label}: adaptive result protocol is not v2")
    if result.get("execution_semantics") != "selective_second_pass_rerun":
        raise SystemExit(f"{label}: adaptive result execution semantics drifted")
    if int(result.get("n_test", -1)) != len(expected_ids):
        raise SystemExit(f"{label}: adaptive result test count is incomplete")
    expected_fraction = float(evaluation["max_refine_fraction"])
    if not np.isclose(float(result["max_refine_fraction"]), expected_fraction):
        raise SystemExit(f"{label}: result/refine-fraction mismatch")

    thresholds = {float(row["threshold"]) for row in rows}
    if len(thresholds) != 1:
        raise SystemExit(f"{label}: prediction file mixes thresholds")
    threshold = next(iter(thresholds))
    if not np.isclose(
        threshold,
        float(result["selected_on_calibration"]["threshold"]),
    ):
        raise SystemExit(f"{label}: prediction/result threshold mismatch")

    row_fractions = {float(row["max_refine_fraction"]) for row in rows}
    if row_fractions != {expected_fraction}:
        raise SystemExit(f"{label}: prediction file mixes refine caps")

    semantics = {str(row["execution_semantics"]) for row in rows}
    if semantics != {"selective_second_pass_rerun"}:
        raise SystemExit(f"{label}: unexpected execution semantics {semantics}")

    selected = np.asarray([bool(row["selected"]) for row in rows], dtype=bool)
    observed_fraction = float(selected.mean())
    if not np.isclose(
        observed_fraction,
        float(result["test"]["refine_fraction"]),
    ):
        raise SystemExit(f"{label}: prediction/result selected fraction mismatch")

    base_budgets = {float(row["base_compute"]) for row in rows}
    refined_budgets = {float(row["refined_compute"]) for row in rows}
    if len(base_budgets) != 1 or len(refined_budgets) != 1:
        raise SystemExit(f"{label}: mixed compute budgets in predictions")

    return AdaptiveCell(
        label=label,
        adapter=str(spec["adapter"]),
        dataset=str(spec["dataset"]),
        base_correct=np.asarray(
            [bool(row["base_correct"]) for row in rows],
            dtype=float,
        ),
        refined_correct=np.asarray(
            [bool(row["refined_correct"]) for row in rows],
            dtype=float,
        ),
        selected=selected,
        base_compute=next(iter(base_budgets)),
        refined_compute=next(iter(refined_budgets)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Hierarchical equal-cell-weight aggregation for frozen adaptive "
            "second-pass experiments."
        )
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--fraction-key", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if plan.get("status") != "PASS":
        raise SystemExit("Adaptive plan is not PASS.")
    provenance_failures = validate_plan_provenance(plan)
    if provenance_failures:
        raise SystemExit(
            "Adaptive plan provenance check failed:\n- "
            + "\n- ".join(provenance_failures)
        )

    cells_spec = plan.get("cells", [])
    expected = int(plan.get("expected_cells", 0))
    if expected <= 0 or len(cells_spec) != expected:
        raise SystemExit(
            f"Adaptive plan cell count mismatch: {len(cells_spec)} != {expected}"
        )

    allowed_keys = {
        str(item["key"]) for item in plan.get("fractions", [])
    }
    if args.fraction_key not in allowed_keys:
        raise SystemExit(
            f"Unknown fraction key {args.fraction_key}; expected {sorted(allowed_keys)}"
        )

    pairs = [(str(item["adapter"]), str(item["dataset"])) for item in cells_spec]
    if len(pairs) != len(set(pairs)):
        raise SystemExit("Adaptive plan contains duplicate model-dataset cells.")

    experiment_commit = str(plan.get("experiment_git_commit", ""))
    if not experiment_commit:
        raise SystemExit("Adaptive plan has no frozen experiment commit.")
    cells = [
        load_prediction_cell(
            spec,
            args.fraction_key,
            experiment_commit,
        )
        for spec in cells_spec
    ]
    result = hierarchical_macro_adaptive(
        cells,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    fraction_info = next(
        item for item in plan["fractions"]
        if str(item["key"]) == args.fraction_key
    )
    provenance_sources: dict[str, str] = {"plan": args.plan}
    for spec in cells_spec:
        label = str(spec["cell"])
        evaluation = spec["evaluations"][args.fraction_key]
        provenance_sources[f"refined_scored:{label}"] = str(
            spec["provenance"]["test"]["refined_scored"]
        )
        provenance_sources[f"adaptive_result:{label}"] = str(
            evaluation["result"]
        )
        provenance_sources[f"adaptive_predictions:{label}"] = str(
            evaluation["predictions"]
        )

    payload = {
        "protocol": "trajuq_adaptive_macro_v1",
        "execution_semantics": "selective_second_pass_rerun",
        "fraction_key": args.fraction_key,
        "max_refine_fraction": float(fraction_info["value"]),
        "is_primary_fraction": bool(fraction_info["primary"]),
        "plan": args.plan,
        "expected_cells": expected,
        "observed_cells": len(cells),
        "cell_set_complete": len(cells) == expected,
        "provenance": analysis_provenance(provenance_sources),
        **result,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
