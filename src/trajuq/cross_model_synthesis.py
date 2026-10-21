from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


AVAILABLE = "available"
PENDING = "pending"
NOT_APPLICABLE = "not_applicable"
VALID_STATES = {AVAILABLE, PENDING, NOT_APPLICABLE}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_value(
    state: str,
    value: float | int | str | None = None,
    *,
    reason: str | None = None,
    source: str | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    interval: str | None = None,
) -> dict[str, Any]:
    if state not in VALID_STATES:
        raise ValueError(f"invalid evidence state: {state}")
    if state == AVAILABLE and value is None:
        raise ValueError("available evidence requires a value")
    if state != AVAILABLE and value is not None:
        raise ValueError(f"{state} evidence may not carry a value")
    if (ci_low is None) != (ci_high is None):
        raise ValueError("confidence interval bounds must be both present or both absent")
    item: dict[str, Any] = {"state": state, "value": value}
    if reason:
        item["reason"] = reason
    if source:
        item["source"] = source
    if ci_low is not None:
        item.update({"ci_low": ci_low, "ci_high": ci_high, "interval": interval})
    return item


def verify_primary_sources(root: Path, expected_commit: str) -> tuple[dict, dict]:
    macro_path = root / "results" / "primary_analysis" / "macro_primary.json"
    gate_path = root / "results" / "primary_analysis" / "rq1_claim_gate.json"
    macro = load_json(macro_path)
    gate = load_json(gate_path)
    if not macro.get("cell_set_complete") or macro.get("observed_cell_count") != 18:
        raise ValueError("RQ1 primary macro is not the complete frozen 18-cell set")
    provenance = gate.get("provenance", {})
    if provenance.get("git_commit") != expected_commit:
        raise ValueError("RQ1 claim gate is bound to a different repository commit")
    expected_hash = (
        provenance.get("sources", {}).get("macro_primary", {}).get("sha256")
    )
    if expected_hash != sha256_file(macro_path):
        raise ValueError("RQ1 macro hash differs from the claim-gate binding")
    return macro, gate


def source_binding(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        return {"path": relative_path, "state": PENDING, "sha256": None}
    return {
        "path": relative_path,
        "state": AVAILABLE,
        "sha256": sha256_file(path),
    }


def required_source_binding(
    root: Path, rq: str, relative_path: str, expected_commit: str
) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        return {"path": relative_path, "state": PENDING, "sha256": None}
    binding = {
        "path": relative_path,
        "state": PENDING,
        "sha256": sha256_file(path),
    }
    try:
        payload = load_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        binding["reason"] = f"invalid JSON evidence: {type(exc).__name__}"
        return binding
    if not isinstance(payload, dict):
        binding["reason"] = "evidence root is not an object"
        return binding
    if rq == "rq1":
        valid = (
            payload.get("decision")
            in {"SUPPORT_POSITIVE_RQ1_CLAIM", "DO_NOT_SUPPORT_POSITIVE_RQ1_CLAIM"}
            and payload.get("provenance", {}).get("git_commit") == expected_commit
        )
    elif rq in {"rq2", "rq3"}:
        complete = payload.get("cell_set_complete") is True
        counts_match = payload.get("observed_cell_count") == payload.get(
            "expected_cell_count"
        )
        valid = complete and counts_match
    else:
        commit = payload.get("authoritative_commit") or payload.get("experiment_commit")
        final_flag = payload.get("final") is True or payload.get("complete") is True
        valid = payload.get("status") in {"PASS", "FINAL"} and final_flag
        if commit is not None:
            valid = valid and commit == expected_commit
    if valid:
        binding["state"] = AVAILABLE
    else:
        binding["reason"] = "authoritative completion/finalization gates are not satisfied"
    return binding


def _parameter_map(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "configs" / "operations" / "model_parameter_registry.json"
    registry = load_json(path)
    return {item["key"]: item for item in registry["models"]}


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _primary_family_metrics(macro: dict[str, Any]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for cell in macro["per_cell"].values():
        family = cell["model_family"]
        for metric in ("auroc", "auprc", "aurc", "ece15", "risk50"):
            grouped[family][metric].append(float(cell["metrics"][metric]["candidate"]))
            grouped[family][f"{metric}_baseline"].append(
                float(cell["metrics"][metric]["baseline"])
            )
            grouped[family][f"{metric}_improvement"].append(
                float(cell["metrics"][metric]["improvement"])
            )
    result: dict[str, dict[str, float]] = {}
    for family, metrics in grouped.items():
        result[family] = {name: _mean(values) for name, values in metrics.items()}
        expected = float(
            macro["macro"]["auroc"]["model_family_macro_improvement"][family]
        )
        if abs(result[family]["auroc_improvement"] - expected) > 1e-12:
            raise ValueError(f"family AUROC aggregation mismatch for {family}")
    return result


def _availability_text(value: str) -> str:
    if value not in VALID_STATES:
        raise ValueError(f"invalid signal availability: {value}")
    return value


def _pending_metrics(reason: str) -> dict[str, dict[str, Any]]:
    return {
        key: evidence_value(PENDING, reason=reason)
        for key in (
            "auroc",
            "auprc",
            "aurc",
            "ece15",
            "risk50",
            "within_regime_auroc_improvement",
        )
    }


def _diffusion_metrics(
    family: str,
    family_metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    source = "results/primary_analysis/macro_primary.json"
    values = family_metrics[family]
    return {
        "auroc": evidence_value(AVAILABLE, values["auroc"], source=source),
        "auprc": evidence_value(AVAILABLE, values["auprc"], source=source),
        "aurc": evidence_value(AVAILABLE, values["aurc"], source=source),
        "ece15": evidence_value(AVAILABLE, values["ece15"], source=source),
        "risk50": evidence_value(AVAILABLE, values["risk50"], source=source),
        "within_regime_auroc_improvement": evidence_value(
            AVAILABLE,
            values["auroc_improvement"],
            source=source,
            reason=(
                "Equal-dataset-weight point estimate. Family-specific interval is not "
                "present in the authoritative aggregate and is therefore not invented."
            ),
        ),
    }


def _appendix_rows(
    contract: dict[str, Any],
    macro: dict[str, Any],
    models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {row["key"]: row for row in models}
    cell_map = {
        (cell["model_family"], cell["dataset"]): cell
        for cell in macro["per_cell"].values()
    }
    rows: list[dict[str, Any]] = []
    for key in contract["model_order"]:
        model = by_key[key]
        for dataset in contract["dataset_order"]:
            cell = cell_map.get((key, dataset))
            if cell is None:
                metrics = _pending_metrics(
                    "Contributing authoritative RQ registry is incomplete or unavailable."
                )
                n_test = evidence_value(PENDING, reason="Authoritative cell unavailable.")
            else:
                metrics = {
                    metric: evidence_value(
                        AVAILABLE,
                        float(cell["metrics"][metric]["candidate"]),
                        source=(
                            f"results/primary_analysis/{key}/{dataset}/metrics.json"
                        ),
                    )
                    for metric in ("auroc", "auprc", "aurc", "ece15", "risk50")
                }
                metrics["within_regime_auroc_improvement"] = evidence_value(
                    AVAILABLE,
                    float(cell["metrics"]["auroc"]["improvement"]),
                    source=f"results/primary_analysis/{key}/{dataset}/metrics.json",
                )
                n_test = evidence_value(AVAILABLE, int(cell["n"]))
            rows.append(
                {
                    "model_key": key,
                    "model": model["display_name"],
                    "regime": model["regime"],
                    "compatibility_group": model["compatibility_group"],
                    "dataset": dataset,
                    "n_test": n_test,
                    "metrics": metrics,
                }
            )
    return rows


def build_registry(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_json(contract_path)
    expected_commit = contract["authoritative_commit"]
    macro, gate = verify_primary_sources(root, expected_commit)
    parameters = _parameter_map(root)
    family_metrics = _primary_family_metrics(macro)

    source_bindings = {
        rq: required_source_binding(root, rq, path, expected_commit)
        for rq, path in contract["required_final_sources"].items()
    }
    rq5_manifest_path = root / "results" / "rq5" / "validated" / "registry_manifest.json"
    if rq5_manifest_path.is_file():
        try:
            rq5_manifest = load_json(rq5_manifest_path)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            source_bindings["rq5_validated_registry"] = {
                "path": str(rq5_manifest_path.relative_to(root)).replace("\\", "/"),
                "state": PENDING,
                "sha256": sha256_file(rq5_manifest_path),
                "partial": True,
                "reason": f"invalid registry JSON: {type(exc).__name__}",
            }
        else:
            source_bindings["rq5_validated_registry"] = {
                "path": str(rq5_manifest_path.relative_to(root)).replace("\\", "/"),
                "state": AVAILABLE,
                "sha256": sha256_file(rq5_manifest_path),
                "partial": bool(rq5_manifest.get("partial", True)),
                "validated_responses": rq5_manifest.get("validated_responses"),
                "expected_responses": rq5_manifest.get("expected_responses"),
            }

    models: list[dict[str, Any]] = []
    for key in contract["model_order"]:
        parameter = parameters[key]
        model_contract = contract["models"][key]
        regime_key = model_contract["regime"]
        regime = contract["regimes"][regime_key]
        if key in family_metrics:
            metrics = _diffusion_metrics(key, family_metrics)
            empirical_state = AVAILABLE
        else:
            required = "rq4" if key == "llama31" else "rq5"
            metrics = _pending_metrics(
                f"{required.upper()} final cross-model aggregate is not authoritative yet."
            )
            empirical_state = PENDING
        total_parameters = parameter.get("total_parameters")
        if total_parameters is None:
            parameter_value = evidence_value(
                NOT_APPLICABLE,
                reason="Provider-managed parameter count is not disclosed or verifiable.",
            )
        else:
            parameter_value = evidence_value(
                AVAILABLE,
                int(total_parameters),
                source="configs/operations/model_parameter_registry.json",
            )
        models.append(
            {
                "key": key,
                "display_name": parameter["display_name"],
                "architecture": parameter["architecture_class"],
                "regime": regime_key,
                "regime_display": regime["display_name"],
                "compatibility_group": regime["compatibility_group"],
                "parameter_count": parameter_value,
                "uncertainty_signal": model_contract["uncertainty_signal"],
                "revision_signal": _availability_text(model_contract["revision_signal"]),
                "persistence_signal": _availability_text(
                    model_contract["persistence_signal"]
                ),
                "commitment_signal": _availability_text(
                    model_contract["commitment_signal"]
                ),
                "compute_or_api_cost": evidence_value(
                    PENDING,
                    reason="Awaiting a model-bound authoritative compute/API ledger aggregate.",
                ),
                "metrics": metrics,
                "empirical_state": empirical_state,
            }
        )

    compatibility = []
    for regime_key, regime in contract["regimes"].items():
        compatibility.append(
            {
                "regime": regime_key,
                **regime,
                "cross_regime_use": "descriptive_only",
                "statistical_ranking_across_regimes": False,
            }
        )

    required_states = {
        rq: source_bindings[rq]["state"] for rq in contract["required_final_sources"]
    }
    final = all(value == AVAILABLE for value in required_states.values())
    if source_bindings.get("rq5_validated_registry", {}).get("partial", True):
        final = False

    registry = {
        "schema": "trajuq_cross_model_synthesis_v1",
        "status": "FINAL" if final else "PROVISIONAL",
        "final": final,
        "scope": contract["purpose"],
        "authoritative_commit": expected_commit,
        "claim_policy": {
            "within_regime_comparisons": "allowed_when_compatibility_group_matches",
            "cross_regime_comparisons": "descriptive_only",
            "causal_architecture_claims": "forbidden",
            "raw_cross_protocol_ranking": "forbidden",
        },
        "models": models,
        "appendix_rows": _appendix_rows(contract, macro, models),
        "compatibility": compatibility,
        "source_bindings": source_bindings,
        "finalization_gate": {
            "required_states": required_states,
            "rq5_registry_reconciled": not source_bindings.get(
                "rq5_validated_registry", {}
            ).get("partial", True),
            "rq1_claim_gate_decision": gate["decision"],
            "pass": final,
        },
    }
    return registry


def assert_compatible_for_direct_comparison(
    left: dict[str, Any], right: dict[str, Any]
) -> None:
    if left["compatibility_group"] != right["compatibility_group"]:
        raise ValueError(
            "metric compatibility firewall: direct cross-regime ranking is forbidden"
        )

