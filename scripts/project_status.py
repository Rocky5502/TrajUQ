#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def dotted_get(payload: dict[str, Any], key: str) -> Any:
    value: Any = payload
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(key)
        value = value[part]
    return value


def implementation_state(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    evidence = [str(path) for path in item.get("evidence", [])]
    missing = [path for path in evidence if not (root / path).exists()]
    declared = str(item.get("state", "UNKNOWN"))
    verified = declared == "DONE" and not missing
    return {
        "id": item["id"],
        "label": item["label"],
        "declared_state": declared,
        "verified": verified,
        "evidence": evidence,
        "missing_evidence": missing,
    }


def gate_state(root: Path, gate: dict[str, Any]) -> dict[str, Any]:
    path = root / str(gate["artifact"])
    base = {
        "id": gate["id"],
        "label": gate["label"],
        "stage": gate.get("stage", "unknown"),
        "issue": gate.get("issue"),
        "artifact": str(gate["artifact"]),
    }
    if not path.exists():
        return {**base, "state": "PENDING", "reason": "artifact_missing"}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {**base, "state": "INVALID", "reason": f"{type(exc).__name__}: {exc}"}

    expect = gate.get("expect")
    if not expect:
        return {**base, "state": "PASS", "reason": "artifact_exists"}

    key = str(expect["key"])
    try:
        observed = dotted_get(payload, key)
    except KeyError:
        return {**base, "state": "FAIL", "reason": f"missing_key:{key}"}

    if "equals" in expect:
        expected = expect["equals"]
        passed = observed == expected
        reason = f"{key}={observed!r}; expected={expected!r}"
    elif "in" in expect:
        allowed = list(expect["in"])
        passed = observed in allowed
        reason = f"{key}={observed!r}; allowed={allowed!r}"
    else:
        return {**base, "state": "INVALID", "reason": "unsupported_expectation"}

    return {
        **base,
        "state": "PASS" if passed else "FAIL",
        "reason": reason,
    }


def build_snapshot(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    implementation = [
        implementation_state(root, item)
        for item in config.get("implementation", [])
    ]
    gates = [
        gate_state(root, gate)
        for gate in config.get("execution_gates", [])
    ]

    verified_impl = sum(item["verified"] for item in implementation)
    passed_gates = sum(gate["state"] == "PASS" for gate in gates)
    failed_gates = sum(gate["state"] in {"FAIL", "INVALID"} for gate in gates)

    return {
        "protocol": "trajuq_project_status_v1",
        "project": config.get("project", {}),
        "implementation": implementation,
        "execution_gates": gates,
        "summary": {
            "implementation_verified": verified_impl,
            "implementation_total": len(implementation),
            "implementation_fraction": (
                verified_impl / len(implementation) if implementation else 0.0
            ),
            "execution_passed": passed_gates,
            "execution_total": len(gates),
            "execution_fraction": passed_gates / len(gates) if gates else 0.0,
            "execution_failed_or_invalid": failed_gates,
            "execution_pending": sum(gate["state"] == "PENDING" for gate in gates),
        },
    }


def render_markdown(snapshot: dict[str, Any]) -> str:
    project = snapshot["project"]
    summary = snapshot["summary"]
    lines = [
        "# TrajUQ local status snapshot",
        "",
        f"Target: **{project.get('target', 'unknown')}**",
        "",
        (
            "Implementation evidence verified: "
            f"**{summary['implementation_verified']}/{summary['implementation_total']}**"
        ),
        (
            "Execution gates passed: "
            f"**{summary['execution_passed']}/{summary['execution_total']}** "
            f"(pending: {summary['execution_pending']}, "
            f"failed/invalid: {summary['execution_failed_or_invalid']})"
        ),
        "",
        "## Implementation",
        "",
        "| Item | Declared | Evidence |",
        "|---|---|---|",
    ]
    for item in snapshot["implementation"]:
        evidence = "OK" if item["verified"] else "MISSING: " + ", ".join(
            item["missing_evidence"]
        )
        lines.append(
            f"| {item['label']} | {item['declared_state']} | {evidence} |"
        )

    lines += [
        "",
        "## Execution gates",
        "",
        "| Stage | Gate | State | Artifact |",
        "|---|---|---|---|",
    ]
    for gate in snapshot["execution_gates"]:
        lines.append(
            f"| {gate['stage']} | {gate['label']} | {gate['state']} | "
            f"{gate['artifact']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit TrajUQ implementation evidence and experiment gate artifacts."
    )
    parser.add_argument(
        "--config",
        default="configs/project_status.yaml",
    )
    parser.add_argument(
        "--json-output",
        default="results/project_status_snapshot.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="results/project_status_snapshot.md",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if declared implementation evidence is missing or a present "
            "execution gate is FAIL/INVALID."
        ),
    )
    args = parser.parse_args()

    root = Path.cwd()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    snapshot = build_snapshot(root, config)

    json_out = Path(args.json_output)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    markdown_out = Path(args.markdown_output)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.write_text(render_markdown(snapshot) + "\n", encoding="utf-8")

    print(render_markdown(snapshot))
    print(f"[TrajUQ] JSON status -> {json_out}")
    print(f"[TrajUQ] Markdown status -> {markdown_out}")

    missing_impl = [
        item for item in snapshot["implementation"] if not item["verified"]
    ]
    bad_gates = [
        gate
        for gate in snapshot["execution_gates"]
        if gate["state"] in {"FAIL", "INVALID"}
    ]
    if args.strict and (missing_impl or bad_gates):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
