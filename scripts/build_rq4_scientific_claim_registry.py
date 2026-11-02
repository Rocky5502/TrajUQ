#!/usr/bin/env python
"""Build the scientific RQ4 claim registry without changing the mechanical manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def build(root: Path) -> dict:
    analysis = root / "results/rq4/analysis"
    paths = {
        "hypothesis_gate": analysis / "hypothesis_gate.json",
        "macro_aggregate": analysis / "macro_aggregate.json",
        "paired_bootstrap": analysis / "paired_bootstrap.json",
        "multiplicity": analysis / "multiplicity.json",
        "finalization_state": analysis / "finalization_state.json",
        "mechanical_manifest": root / "results/generated/final/rq4_manifest.json",
        "downstream_plan": root / "results/rq4/rq4_downstream_plan.json",
    }
    documents = {name: read_json(path) for name, path in paths.items()}
    gate = documents["hypothesis_gate"]
    macro = documents["macro_aggregate"]
    bootstrap = documents["paired_bootstrap"]
    multiplicity = documents["multiplicity"]
    state = documents["finalization_state"]
    mechanical = documents["mechanical_manifest"]

    plan_hash = sha256(paths["downstream_plan"])
    sidecar = Path(str(paths["downstream_plan"]) + ".sha256").read_text(encoding="utf-8").strip()
    if plan_hash != sidecar or plan_hash != state["plan_sha256"]:
        raise SystemExit("RQ4 downstream-plan hash drift")
    if sha256(paths["mechanical_manifest"]) != state["final_manifest"]["sha256"]:
        raise SystemExit("RQ4 mechanical manifest hash drift")
    if mechanical.get("claim_authorized") is not False:
        raise SystemExit("RQ4 mechanical claim firewall was modified")
    if gate.get("status") != "PASS" or gate.get("claim_authorized") is not True:
        raise SystemExit("RQ4 scientific gate is not authorized")
    if state.get("status") != "COMPLETE":
        raise SystemExit("RQ4 downstream state is incomplete")
    generation_hash = state["generation_manifest_sha256"]
    if generation_hash != mechanical["sources"][0]["sha256"]:
        raise SystemExit("RQ4 generation-manifest binding drift")

    stage_hashes = {
        output["path"]: output["sha256"]
        for stage in state["completed_stages"]
        for output in stage["outputs"]
    }
    for name in ("hypothesis_gate", "macro_aggregate", "paired_bootstrap", "multiplicity"):
        relative = paths[name].relative_to(root).as_posix()
        if stage_hashes.get(relative) != sha256(paths[name]):
            raise SystemExit(f"RQ4 stage hash drift: {relative}")

    claims = {}
    for hypothesis in ("h4a", "h4b", "h4c"):
        decision = gate["decisions"][hypothesis]
        interval = bootstrap["hypotheses"][hypothesis]
        correction = multiplicity["decisions"][hypothesis]
        if decision["estimate"] != interval["estimate"]:
            raise SystemExit(f"RQ4 estimate disagreement for {hypothesis}")
        if decision["adjusted_p"] != correction["adjusted_p"]:
            raise SystemExit(f"RQ4 multiplicity disagreement for {hypothesis}")
        claims[hypothesis] = {
            "decision": decision["decision"],
            "estimate": decision["estimate"],
            "ci_95": [interval["ci_low"], interval["ci_high"]],
            "raw_p_two_sided": interval["p_value_two_sided"],
            "holm_adjusted_p": decision["adjusted_p"],
        }

    reference_names = ("mean_token_entropy", "trajectory", "five_sample_semantic_disagreement")
    registry = {
        "schema": "trajuq_rq4_scientific_claim_registry_v1",
        "status": "PASS",
        "claim_authorized": True,
        "mechanical_manifest_claim_authorized": False,
        "chronology": "confirmatory protocol pre-generation; orchestration plan post-generation/pre-outcome-scoring",
        "generation_manifest_sha256": generation_hash,
        "downstream_plan_sha256": plan_hash,
        "analysis_commit": "4b45dbb9c50284f108ea56aa8b90e2aebbd9babf",
        "bindings": [
            {
                "role": name,
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
            }
            for name, path in paths.items()
        ],
        "confirmatory_claims": claims,
        "macro_reference": {
            name: {
                "aurc": macro["macro"][name]["aurc"],
                "auroc": macro["macro"][name]["auroc"],
            }
            for name in reference_names
        },
        "authorized_interpretation": [
            "Trajectory-aware uncertainty improves selective-risk ranking over mean token entropy.",
            "Five-sample semantic disagreement is stronger in the reported macro comparison but requires multiple generations.",
        ],
        "prohibited_interpretation": [
            "Do not treat the mechanical rq4_manifest.json as a scientific claim gate.",
            "Do not claim that trajectory uncertainty itself is novel.",
            "Do not generalize beyond the registered datasets, model revision, and observability regime.",
        ],
    }
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/generated/final/rq4_scientific_claim_registry.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    registry = build(root)
    payload = (json.dumps(registry, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output = args.output if args.output.is_absolute() else root / args.output
    write_new(output, payload)
    write_new(Path(str(output) + ".sha256"), (hashlib.sha256(payload).hexdigest() + "\n").encode())


if __name__ == "__main__":
    main()
