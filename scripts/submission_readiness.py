#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_overleaf_package import (  # noqa: E402
    ACL_STYLE_COMMIT,
    PAPER_FILES,
    REQUIRED_RESULT_FIGURES,
)
from scripts.execute_adaptive_plan import (  # noqa: E402
    validate_plan_provenance as validate_adaptive_provenance,
)
from scripts.execute_primary_analysis import (  # noqa: E402
    validate_plan_provenance as validate_analysis_provenance,
)
from scripts.execute_primary_plan import (  # noqa: E402
    validate_plan_provenance as validate_primary_provenance,
)
from scripts.execute_repeated_sampling import (  # noqa: E402
    validate_plan_provenance as validate_repeated_provenance,
)
from scripts.project_status import build_snapshot  # noqa: E402
from scripts.render_paper_result_figures import (  # noqa: E402
    require_complete_adaptive as require_complete_figure_adaptive,
    require_complete_primary as require_complete_figure_primary,
)
from scripts.render_results_tex import (  # noqa: E402
    build_macros,
    render_tex,
    require_bound_result_provenance,
    require_render_source_provenance,
)


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


def preprimary_revalidation_argv() -> list[str]:
    return [
        sys.executable,
        "scripts/primary_readiness.py",
        "--historical-provenance-only",
        "--output",
        "results/submission_preprimary_revalidation.json",
    ]


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
    repeated_plan_path: Path | None = None,
    adaptive_plan_path: Path | None = None,
) -> dict:
    manifest = load_json(manifest_path)
    failures: list[str] = []

    if manifest.get("protocol") != "trajuq_paper_results_render_v1":
        failures.append("generated-results manifest protocol is not recognized")

    sources = manifest.get("sources", {})
    required_sources = {
        "primary",
        "claim_gate",
        "repeated",
        "ablation",
        "prior_signals",
        "adaptive",
        "primary_config",
    }
    if not isinstance(sources, dict):
        failures.append("generated-results manifest sources is not an object")
        sources = {}
    missing_sources = sorted(required_sources - set(sources))
    if missing_sources:
        failures.append(
            "generated-results manifest missing required sources: "
            + ", ".join(missing_sources)
        )

    decision = manifest.get("rq1_claim_decision")
    if decision not in {
        "SUPPORT_POSITIVE_RQ1_CLAIM",
        "DO_NOT_MAKE_POSITIVE_RQ1_CLAIM",
    }:
        failures.append("generated-results manifest has no valid RQ1 claim decision")
    try:
        if int(manifest.get("macro_count", 0)) <= 0:
            failures.append("generated-results manifest has no rendered macros")
    except (TypeError, ValueError):
        failures.append("generated-results manifest macro_count is invalid")

    source_paths: dict[str, Path] = {}
    source_hashes_valid = True
    for name, source in sources.items():
        path = Path(str(source["path"]))
        source_paths[name] = path
        if not path.exists():
            source_hashes_valid = False
            failures.append(f"render source missing: {name} -> {path}")
            continue
        observed = sha256_file(path)
        expected = str(source.get("sha256"))
        if observed != expected:
            source_hashes_valid = False
            failures.append(
                f"render source hash changed: {name} {observed} != {expected}"
            )

    expected_rendered_text: str | None = None
    if (
        adaptive_plan_path is not None
        and required_sources.issubset(source_paths)
        and source_hashes_valid
    ):
        try:
            primary = load_json(source_paths["primary"])
            claim = load_json(source_paths["claim_gate"])
            repeated = load_json(source_paths["repeated"])
            ablation = load_json(source_paths["ablation"])
            prior_signals = load_json(source_paths["prior_signals"])
            adaptive = load_json(source_paths["adaptive"])
            primary_cfg = yaml.safe_load(
                source_paths["primary_config"].read_text(encoding="utf-8")
            )
            render_payloads = {
                "primary": primary,
                "claim_gate": claim,
                "repeated": repeated,
                "ablation": ablation,
                "prior_signals": prior_signals,
                "adaptive": adaptive,
            }
            require_render_source_provenance(
                render_payloads,
                source_paths,
                source_paths["primary_config"],
            )
            for label, plan_path in (
                ("repeated", repeated_plan_path),
                ("adaptive", adaptive_plan_path),
            ):
                if plan_path is None:
                    continue
                if not plan_path.is_file():
                    raise ValueError(
                        f"{label} frozen plan is missing: {plan_path}"
                    )
                plan_source = render_payloads[label].get(
                    "provenance", {}
                ).get("sources", {}).get("plan")
                if not isinstance(plan_source, dict):
                    raise ValueError(
                        f"{label} macro provenance does not bind its frozen plan."
                    )
                if (
                    str(Path(str(plan_source.get("path", ""))))
                    != str(plan_path)
                    or str(plan_source.get("sha256", ""))
                    != sha256_file(plan_path)
                ):
                    raise ValueError(
                        f"{label} macro provenance binds a different frozen plan."
                    )
            macros = build_macros(
                primary,
                claim,
                repeated,
                ablation,
                prior_signals,
                adaptive,
                primary_cfg,
            )
            expected_sources = {
                name: (
                    str(source_paths[name]),
                    str(sources[name]["sha256"]),
                )
                for name in (
                    "primary",
                    "claim_gate",
                    "repeated",
                    "ablation",
                    "prior_signals",
                    "adaptive",
                    "primary_config",
                )
            }
            expected_rendered_text = render_tex(macros, expected_sources)
            if int(manifest.get("macro_count", -1)) != len(macros):
                failures.append(
                    "generated-results manifest macro_count differs from "
                    "deterministic re-render"
                )
            if manifest.get("rq1_claim_decision") != claim.get("decision"):
                failures.append(
                    "generated-results manifest RQ1 decision differs from "
                    "claim-gate source"
                )
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            failures.append(
                f"render source semantic revalidation failed: {exc}"
            )

    if not generated_tex.exists():
        failures.append(f"missing generated result macros: {generated_tex}")
    else:
        observed_tex_sha = sha256_file(generated_tex)
        if observed_tex_sha != str(manifest.get("output_sha256")):
            failures.append("generated_results.tex hash differs from render manifest")

        text = generated_tex.read_text(encoding="utf-8")
        if (
            expected_rendered_text is not None
            and text != expected_rendered_text
        ):
            failures.append(
                "generated_results.tex differs from deterministic re-render "
                "of the frozen source artifacts"
            )
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


def verify_result_figures(
    manifest_path: Path,
    adaptive_plan_path: Path | None = None,
) -> dict:
    failures: list[str] = []
    try:
        manifest = load_json(manifest_path)
    except ValueError as exc:
        return {
            "status": "FAIL",
            "manifest": str(manifest_path),
            "failures": [str(exc)],
        }

    if manifest.get("protocol") != "trajuq_paper_result_figures_v2":
        failures.append("result-figure manifest protocol is not v2")
    try:
        expected_cells = int(manifest.get("expected_cells", -1))
    except (TypeError, ValueError):
        expected_cells = -1
    if expected_cells != 18:
        failures.append("result-figure manifest expected cell count is not 18")

    required_sources = {
        "primary",
        "primary_config",
        "cap_025",
        "cap_050",
        "cap_075",
    }
    sources = manifest.get("sources", {})
    if not isinstance(sources, dict):
        failures.append("result-figure sources is not an object")
        sources = {}
    missing_sources = sorted(required_sources - set(sources))
    if missing_sources:
        failures.append(
            "result-figure manifest missing sources: "
            + ", ".join(missing_sources)
        )

    source_paths: dict[str, Path] = {}
    source_hashes_valid = True
    for name in sorted(required_sources.intersection(sources)):
        source = sources[name]
        if not isinstance(source, dict):
            source_hashes_valid = False
            failures.append(f"result-figure source entry is invalid: {name}")
            continue
        path = Path(str(source.get("path", "")))
        source_paths[name] = path
        if not path.exists():
            source_hashes_valid = False
            failures.append(f"result-figure source missing: {name} -> {path}")
            continue
        if sha256_file(path) != str(source.get("sha256", "")):
            source_hashes_valid = False
            failures.append(f"result-figure source hash changed: {name}")

    if required_sources.issubset(source_paths) and source_hashes_valid:
        try:
            primary = load_json(source_paths["primary"])
            cap_payloads = {
                key: load_json(source_paths[key])
                for key in ("cap_025", "cap_050", "cap_075")
            }
            commits = {
                require_bound_result_provenance(
                    primary,
                    label="primary",
                ),
                *[
                    require_bound_result_provenance(
                        payload,
                        label=key,
                    )
                    for key, payload in cap_payloads.items()
                ],
            }
            if len(commits) != 1:
                raise ValueError(
                    "result-figure aggregate sources disagree on the frozen commit."
                )

            require_complete_figure_primary(primary, 18)
            expected_caps = {
                "cap_025": (0.25, False),
                "cap_050": (0.50, True),
                "cap_075": (0.75, False),
            }
            for key, (fraction, is_primary) in expected_caps.items():
                require_complete_figure_adaptive(
                    cap_payloads[key],
                    18,
                    fraction_key=key,
                    expected_fraction=fraction,
                    is_primary=is_primary,
                )

            config_source = primary.get("provenance", {}).get(
                "sources", {}
            ).get("primary_config")
            config_path = source_paths["primary_config"]
            if not isinstance(config_source, dict):
                raise ValueError(
                    "primary figure source does not bind the primary config."
                )
            if (
                str(Path(str(config_source.get("path", ""))))
                != str(config_path)
                or str(config_source.get("sha256", ""))
                != sha256_file(config_path)
            ):
                raise ValueError(
                    "primary figure source binds a different primary config."
                )

            if adaptive_plan_path is not None:
                if not adaptive_plan_path.is_file():
                    raise ValueError(
                        f"adaptive frozen plan is missing: {adaptive_plan_path}"
                    )
                expected_plan_hash = sha256_file(adaptive_plan_path)
                for key, payload in cap_payloads.items():
                    plan_source = payload.get(
                        "provenance", {}
                    ).get("sources", {}).get("plan")
                    if not isinstance(plan_source, dict):
                        raise ValueError(
                            f"{key} provenance does not bind the adaptive plan."
                        )
                    if (
                        str(Path(str(plan_source.get("path", ""))))
                        != str(adaptive_plan_path)
                        or str(plan_source.get("sha256", ""))
                        != expected_plan_hash
                    ):
                        raise ValueError(
                            f"{key} provenance binds a different adaptive plan."
                        )
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            failures.append(
                f"result-figure source provenance revalidation failed: {exc}"
            )

    required_outputs = {
        "primary_auroc_matrix",
        "family_macro_auroc",
        "adaptive_tradeoff",
    }
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        failures.append("result-figure outputs is not an object")
        outputs = {}
    missing_outputs = sorted(required_outputs - set(outputs))
    if missing_outputs:
        failures.append(
            "result-figure manifest missing outputs: "
            + ", ".join(missing_outputs)
        )

    for name in sorted(required_outputs.intersection(outputs)):
        item = outputs[name]
        if not isinstance(item, dict):
            failures.append(f"result-figure output entry is invalid: {name}")
            continue
        path = Path(str(item.get("path", "")))
        if not path.exists():
            failures.append(f"result figure missing: {name} -> {path}")
            continue
        if path.suffix.lower() != ".pdf":
            failures.append(f"result figure is not a PDF: {name}")
            continue
        data = path.read_bytes()
        if not data.startswith(b"%PDF-"):
            failures.append(f"result figure has invalid PDF header: {name}")
        if sha256_file(path) != str(item.get("sha256", "")):
            failures.append(f"result figure hash changed: {name}")
        try:
            expected_bytes = int(item.get("bytes", -1))
        except (TypeError, ValueError):
            expected_bytes = -1
        if expected_bytes != path.stat().st_size:
            failures.append(f"result figure byte count changed: {name}")

    return {
        "status": "PASS" if not failures else "FAIL",
        "manifest": str(manifest_path),
        "failures": failures,
    }


def verify_overleaf_package(
    path: Path,
    *,
    paper_dir: Path | None = None,
    result_figure_manifest: Path | None = None,
) -> dict:
    failures: list[str] = []
    if not path.is_file():
        return {
            "status": "FAIL",
            "path": str(path),
            "exists": False,
            "sha256": None,
            "bytes": None,
            "failures": [f"anonymous Overleaf package is missing: {path}"],
        }

    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            if "MANIFEST.json" not in names:
                raise ValueError("Overleaf ZIP has no MANIFEST.json")
            manifest = json.loads(
                zf.read("MANIFEST.json").decode("utf-8")
            )
            if manifest.get("protocol") != "trajuq_overleaf_package_v1":
                failures.append("Overleaf package protocol is not recognized")
            if manifest.get("anonymous") is not True:
                failures.append("Overleaf package is not marked anonymous")
            if manifest.get("submission_ready") is not True:
                failures.append("Overleaf package is not submission-ready")
            if manifest.get("acl_style_commit") != ACL_STYLE_COMMIT:
                failures.append("Overleaf package ACL style commit drifted")

            required_names = {
                *PAPER_FILES,
                "acl.sty",
                "acl_natbib.bst",
                "PAPER_STATUS.md",
                "MANIFEST.json",
                "figures/generated/primary_auroc_improvement_matrix.pdf",
                "figures/generated/family_macro_auroc_improvement.pdf",
                "figures/generated/adaptive_accuracy_cost_tradeoff.pdf",
            }
            missing = sorted(required_names - names)
            if missing:
                failures.append(
                    "Overleaf package missing required files: "
                    + ", ".join(missing)
                )

            generated_pdf_names = {
                name
                for name in names
                if name.startswith("figures/generated/")
                and name.lower().endswith(".pdf")
            }
            required_pdf_names = {
                "figures/generated/primary_auroc_improvement_matrix.pdf",
                "figures/generated/family_macro_auroc_improvement.pdf",
                "figures/generated/adaptive_accuracy_cost_tradeoff.pdf",
            }
            if generated_pdf_names != required_pdf_names:
                failures.append(
                    "Overleaf generated-PDF set differs from frozen result figures"
                )

            entries = manifest.get("files")
            if not isinstance(entries, list):
                failures.append("Overleaf package manifest files is not a list")
                entries = []
            by_name = {
                str(item.get("path", "")): item
                for item in entries
                if isinstance(item, dict)
            }
            archive_payload_names = names - {"MANIFEST.json"}
            if set(by_name) != archive_payload_names:
                failures.append(
                    "Overleaf package manifest file set differs from ZIP contents"
                )
            for name in sorted(archive_payload_names.intersection(by_name)):
                data = zf.read(name)
                item = by_name[name]
                if hashlib.sha256(data).hexdigest() != str(
                    item.get("sha256", "")
                ):
                    failures.append(
                        f"Overleaf package entry hash differs: {name}"
                    )
                try:
                    expected_bytes = int(item.get("bytes", -1))
                except (TypeError, ValueError):
                    expected_bytes = -1
                if len(data) != expected_bytes:
                    failures.append(
                        f"Overleaf package entry byte count differs: {name}"
                    )

            if paper_dir is not None:
                for name in PAPER_FILES:
                    current = paper_dir / name
                    if not current.is_file():
                        failures.append(
                            f"current paper source missing for Overleaf check: {current}"
                        )
                        continue
                    if name not in names or zf.read(name) != current.read_bytes():
                        failures.append(
                            f"Overleaf package paper source is stale: {name}"
                        )

            if result_figure_manifest is not None:
                if not result_figure_manifest.is_file():
                    failures.append(
                        "current result-figure manifest missing for Overleaf check"
                    )
                else:
                    figure_manifest = load_json(result_figure_manifest)
                    outputs = figure_manifest.get("outputs", {})
                    if not isinstance(outputs, dict):
                        failures.append(
                            "current result-figure manifest outputs is invalid"
                        )
                    else:
                        for key in sorted(REQUIRED_RESULT_FIGURES):
                            item = outputs.get(key)
                            if not isinstance(item, dict):
                                failures.append(
                                    f"current result-figure manifest missing: {key}"
                                )
                                continue
                            source_path = Path(str(item.get("path", "")))
                            if not source_path.is_file():
                                failures.append(
                                    f"current result figure missing: {key}"
                                )
                                continue
                            archive_name = (
                                "figures/generated/" + source_path.name
                            )
                            if (
                                archive_name not in names
                                or hashlib.sha256(
                                    zf.read(archive_name)
                                ).hexdigest()
                                != str(item.get("sha256", ""))
                            ):
                                failures.append(
                                    f"Overleaf package result figure is stale: {key}"
                                )
    except (
        OSError,
        zipfile.BadZipFile,
        KeyError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        failures.append(f"invalid anonymous Overleaf package: {exc}")

    return {
        "status": "PASS" if not failures else "FAIL",
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "expected_result_figure_keys": sorted(REQUIRED_RESULT_FIGURES),
        "failures": failures,
    }


def prerequisite_gate_failures(
    snapshot: dict,
    *,
    self_gate_id: str = "submission_readiness",
) -> list[dict]:
    """Return unmet required prerequisites, excluding this gate's own output.

    The submission-readiness artifact is produced by this script, so treating
    it as an input prerequisite would make the first valid run impossible.
    """
    return [
        gate
        for gate in snapshot["execution_gates"]
        if gate["required"]
        and gate["id"] != self_gate_id
        and gate["state"] != "PASS"
    ]


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
        "--result-figure-manifest",
        default="paper/figures/generated/result_figure_manifest.json",
    )
    parser.add_argument(
        "--overleaf",
        default="dist/TrajUQ_NAACL2027_ARR_Overleaf.zip",
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
        "--skip-overleaf-build",
        action="store_true",
        help="Check an existing submission-ready Overleaf ZIP instead of rebuilding it.",
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
        preprimary_revalidation_argv()
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
    required_not_pass = prerequisite_gate_failures(snapshot)
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
        Path(args.repeated_plan),
        Path(args.adaptive_plan),
    )
    failures.extend(rendered["failures"])

    result_figures = verify_result_figures(
        Path(args.result_figure_manifest),
        Path(args.adaptive_plan),
    )
    failures.extend(result_figures["failures"])

    paper_audit = run_json_command(
        [sys.executable, "scripts/audit_paper.py", "--submission-ready"]
    )
    if paper_audit.get("status") != "PASS":
        failures.append("submission-ready paper audit failed")

    overleaf_path = Path(args.overleaf)
    overleaf_build = None
    if not args.skip_overleaf_build:
        overleaf_build = run_json_command(
            [
                sys.executable,
                "scripts/build_overleaf_package.py",
                "--submission-ready",
                "--output",
                str(overleaf_path),
            ]
        )
        if overleaf_build.get("status") != "PASS":
            failures.append("submission-ready anonymous Overleaf build failed")

    overleaf = verify_overleaf_package(
        overleaf_path,
        paper_dir=Path("paper"),
        result_figure_manifest=Path(args.result_figure_manifest),
    )
    failures.extend(overleaf["failures"])

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
        "submission_gate_self_excluded_from_prerequisites": "submission_readiness",
        "required_gate_failures": required_not_pass,
        "submission_checkout": submission_checkout,
        "preprimary_readiness_revalidation": preprimary_readiness,
        "downstream_provenance_revalidation": downstream_provenance,
        "generated_results": rendered,
        "result_figures": result_figures,
        "paper_audit": paper_audit,
        "overleaf_build": overleaf_build,
        "overleaf": overleaf,
        "supplement_build": supplement_build,
        "supplement": supplement,
        "remaining_external_checks": [
            "Compile the exact recorded anonymous Overleaf source ZIP.",
            "Run the official ACL pubcheck on the final PDF.",
            "Confirm page limits, anonymity, and submission-form metadata.",
            (
                "Upload the exact Overleaf and anonymous supplement archives "
                "whose SHA-256 values are recorded here."
            ),
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
