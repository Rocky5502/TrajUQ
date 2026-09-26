#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.execute_adaptive_plan import (
    validate_plan_provenance as validate_adaptive_provenance,
)
from scripts.execute_primary_analysis import (
    validate_plan_provenance as validate_analysis_provenance,
)
from scripts.execute_primary_plan import (
    validate_plan_provenance as validate_primary_provenance,
)
from scripts.execute_repeated_sampling import (
    validate_plan_provenance as validate_repeated_provenance,
)
from scripts.project_status import build_snapshot


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"missing artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc


def git_text(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception:
        return None


def verify_submission_checkout() -> dict:
    commit = git_text("rev-parse", "HEAD")
    status = git_text("status", "--porcelain")
    failures: list[str] = []
    if not commit:
        failures.append("current submission git commit could not be resolved")
    if status is None:
        failures.append("current submission worktree status could not be resolved")
        clean = None
    else:
        clean = status == ""
        if not clean:
            failures.append(
                "current submission worktree is dirty; commit/stash final paper changes "
                "before the submission gate"
            )
    return {
        "status": "PASS" if not failures else "FAIL",
        "git_commit": commit,
        "git_clean": clean,
        "failures": failures,
    }


def verify_downstream_provenance(
    primary_plan_path: Path,
    analysis_plan_path: Path,
    repeated_plan_path: Path,
    adaptive_plan_path: Path,
) -> dict:
    validators = (
        ("primary", primary_plan_path, validate_primary_provenance),
        ("analysis", analysis_plan_path, validate_analysis_provenance),
        ("repeated", repeated_plan_path, validate_repeated_provenance),
        ("adaptive", adaptive_plan_path, validate_adaptive_provenance),
    )
    checks: dict[str, dict] = {}
    failures: list[str] = []

    for label, path, validator in validators:
        try:
            payload = load_json(path)
        except ValueError as exc:
            checks[label] = {
                "path": str(path),
                "status": "FAIL",
                "failures": [str(exc)],
            }
            failures.append(f"{label} plan unavailable/invalid: {exc}")
            continue

        plan_failures = validator(
            payload,
            require_current_commit=False,
        )
        checks[label] = {
            "path": str(path),
            "protocol": payload.get("protocol"),
            "status": "PASS" if not plan_failures else "FAIL",
            "experiment_git_commit": payload.get("experiment_git_commit"),
            "failures": plan_failures,
        }
        failures.extend(
            f"{label}: {failure}"
            for failure in plan_failures
        )

    return {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
    }


def verify_generated_results(
    manifest_path: Path,
    generated_tex: Path,
) -> dict:
    manifest = load_json(manifest_path)
    failures: list[str] = []

    if manifest.get("protocol") != "trajuq_paper_results_render_v1":
        failures.append("generated-results manifest protocol is not recognized")

    for name, source in manifest.get("sources", {}).items():
        path = Path(str(source["path"]))
        if not path.exists():
            failures.append(f"render source missing: {name} -> {path}")
            continue
        observed = sha256_file(path)
        expected = str(source.get("sha256"))
        if observed != expected:
            failures.append(
                f"render source hash changed: {name} {observed} != {expected}"
            )

    if not generated_tex.exists():
        failures.append(f"missing generated result macros: {generated_tex}")
    else:
        observed_tex_sha = sha256_file(generated_tex)
        if observed_tex_sha != str(manifest.get("output_sha256")):
            failures.append("generated_results.tex hash differs from render manifest")

        text = generated_tex.read_text(encoding="utf-8")
        forbidden = [
            "PRE-RESULTS FALLBACK",
            r"\tbd",
            "pre-experiment manuscript",
        ]
        remaining = [marker for marker in forbidden if marker in text]
        if remaining:
            failures.append(
                f"pre-results placeholders remain in generated results: {remaining}"
            )

    return {
        "status": "PASS" if not failures else "FAIL",
        "manifest": str(manifest_path),
        "generated_tex": str(generated_tex),
        "failures": failures,
    }


def run_json_command(argv: list[str]) -> dict:
    completed = subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = completed.stdout or ""
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        payload = {
            "status": "FAIL",
            "failures": [
                f"command output was not JSON; exit={completed.returncode}",
                output[-4000:],
            ],
        }
    payload["command"] = argv
    payload["returncode"] = int(completed.returncode)
    if completed.returncode != 0:
        payload["status"] = "FAIL"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Final TrajUQ ARR submission gate. This checks that every required "
            "experiment artifact passed, rendered numbers still match their "
            "source hashes, the anonymous paper audit passes in submission mode, "
            "and the deterministic software supplement builds."
        )
    )
    parser.add_argument(
        "--status-config",
        default="configs/project_status.yaml",
    )
    parser.add_argument(
        "--primary-plan",
        default="results/primary_matrix_plan.json",
    )
    parser.add_argument(
        "--primary-analysis-plan",
        default="results/primary_analysis_plan.json",
    )
    parser.add_argument(
        "--repeated-plan",
        default="results/repeated_sampling_plan.json",
    )
    parser.add_argument(
        "--adaptive-plan",
        default="results/adaptive_matrix_plan.json",
    )
    parser.add_argument(
        "--results-manifest",
        default="paper/generated_results_manifest.json",
    )
    parser.add_argument(
        "--generated-tex",
        default="paper/generated_results.tex",
    )
    parser.add_argument(
        "--supplement",
        default="dist/trajuq_anonymous_software.zip",
    )
    parser.add_argument(
        "--output",
        default="results/submission_readiness.json",
    )
    parser.add_argument(
        "--skip-supplement-build",
        action="store_true",
        help="Check an existing supplement instead of rebuilding it.",
    )
    args = parser.parse_args()

    failures: list[str] = []

    submission_checkout = verify_submission_checkout()
    failures.extend(submission_checkout["failures"])

    preprimary_readiness = run_json_command(
        [
            sys.executable,
            "scripts/primary_readiness.py",
            "--historical-provenance-only",
        ]
    )
    if preprimary_readiness.get("status") != "PASS":
        failures.append(
            "historical pre-primary provenance/readiness revalidation failed "
            "at submission time"
        )

    config = yaml.safe_load(
        Path(args.status_config).read_text(encoding="utf-8")
    )
    snapshot = build_snapshot(Path.cwd(), config)
    required_not_pass = [
        gate
        for gate in snapshot["execution_gates"]
        if gate["required"] and gate["state"] != "PASS"
    ]
    if required_not_pass:
        failures.append(
            "required project gates are not all PASS: "
            + ", ".join(
                f"{gate['id']}={gate['state']}"
                for gate in required_not_pass
            )
        )

    downstream_provenance = verify_downstream_provenance(
        Path(args.primary_plan),
        Path(args.primary_analysis_plan),
        Path(args.repeated_plan),
        Path(args.adaptive_plan),
    )
    if downstream_provenance["status"] != "PASS":
        failures.append(
            "downstream experiment-plan provenance revalidation failed"
        )

    rendered = verify_generated_results(
        Path(args.results_manifest),
        Path(args.generated_tex),
    )
    failures.extend(rendered["failures"])

    paper_audit = run_json_command(
        [sys.executable, "scripts/audit_paper.py", "--submission-ready"]
    )
    if paper_audit.get("status") != "PASS":
        failures.append("submission-ready paper audit failed")

    supplement_path = Path(args.supplement)
    supplement_build = None
    if not args.skip_supplement_build:
        supplement_build = run_json_command(
            [
                sys.executable,
                "scripts/build_anonymous_supplement.py",
                "--output",
                str(supplement_path),
            ]
        )
        if supplement_build.get("status") != "PASS":
            failures.append("anonymous software supplement build failed")

    supplement = {
        "path": str(supplement_path),
        "exists": supplement_path.exists(),
        "sha256": (
            sha256_file(supplement_path)
            if supplement_path.exists()
            else None
        ),
        "bytes": (
            supplement_path.stat().st_size
            if supplement_path.exists()
            else None
        ),
    }
    if not supplement["exists"]:
        failures.append("anonymous software supplement is missing")

    payload = {
        "protocol": "trajuq_submission_readiness_v1",
        "status": "PASS" if not failures else "FAIL",
        "project_gate_summary": snapshot["summary"],
        "required_gate_failures": required_not_pass,
        "submission_checkout": submission_checkout,
        "preprimary_readiness_revalidation": preprimary_readiness,
        "downstream_provenance_revalidation": downstream_provenance,
        "generated_results": rendered,
        "paper_audit": paper_audit,
        "supplement_build": supplement_build,
        "supplement": supplement,
        "remaining_external_checks": [
            "Compile the official ACL/ARR PDF with the current acl.sty.",
            "Run the official ACL pubcheck on the final PDF.",
            "Confirm page limits, anonymity, and submission-form metadata.",
            "Upload the exact anonymous supplement whose SHA-256 is recorded here.",
        ],
        "failures": failures,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
