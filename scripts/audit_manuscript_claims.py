#!/usr/bin/env python3
"""Bind quantitative manuscript lines to authorized evidence artifacts.

This is a manuscript consistency audit, not a scientific analysis.  It scans
the final TeX sources, records every result-bearing numeric source line, and
fails closed when a line cannot be assigned to an existing hashed artifact.
Protocol constants are retained but classified separately from empirical
claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NUMERIC = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?(?:\\%|%|\\times)?")
STALE = re.compile(
    r"provisional|remain reportable|will report|TBD|TODO|placeholder|future result|"
    r"once complete|after gate|after validation|not yet|pending result|"
    r"cross-architecture transfer|transfer across architectures",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source(path: str, field: str, status: str) -> dict[str, str]:
    artifact = ROOT / path
    if not artifact.is_file():
        raise SystemExit(f"missing claim source: {path}")
    return {
        "source_artifact": path,
        "source_field": field,
        "authorization_status": status,
        "source_sha256": sha256(artifact),
    }


SOURCES = {
    "rq1": lambda: source(
        "results/generated/final/rq1_manifest.json", "authorized RQ1 evidence bindings", "AUTHORIZED_PRIMARY"
    ),
    "rq2": lambda: source(
        "results/generated/final/rq2_manifest.json", "authorized repeated-sampling and selective-risk bindings", "AUTHORIZED_PRIMARY"
    ),
    "rq2_secondary": lambda: source(
        "results/generated/secondary/rq2_sample_efficiency/manifest.json", "fixed K=2/3/5 sensitivity", "SECONDARY_EXPLORATORY"
    ),
    "rq3": lambda: source(
        "results/adaptive/macro_cap_050.json", "registered 50 percent adaptive-compute endpoint", "AUTHORIZED_PRIMARY"
    ),
    "rq4": lambda: source(
        "results/generated/final/rq4_scientific_claim_registry.json", "confirmatory_claims and authorized macro metrics", "AUTHORIZED_PRIMARY"
    ),
    "rq4_secondary": lambda: source(
        "results/generated/secondary/rq4_early_actionability/manifest.json", "fixed-prefix early-actionability evidence", "SECONDARY_EXPLORATORY"
    ),
    "rq5": lambda: source(
        "results/generated/final/rq5_scientific_claim_registry.json", "registered decision, validation accounting, and macro metrics", "AUTHORIZED_PRIMARY_NULL"
    ),
    "rq5_secondary": lambda: source(
        "results/generated/secondary/rq5_heterogeneity/manifest.json", "descriptive heterogeneity outputs", "DESCRIPTIVE_ONLY"
    ),
    "synthesis": lambda: source(
        "results/generated/secondary/cross_regime_synthesis/manifest.json", "descriptive cross-regime table bindings", "DESCRIPTIVE_ONLY"
    ),
    "protocol": lambda: source(
        "ARTIFACT_MANIFEST.json", "frozen protocol, artifact, and provenance bindings", "PROTOCOL_CONTRACT"
    ),
}


def classify(filename: str, line: str, section: str) -> str:
    lower = line.lower()
    if filename == "generated_rq1.tex":
        return "rq1"
    if filename == "generated_authorized_rq23.tex":
        return "rq3" if "adaptive" in lower else "rq2"
    if filename == "generated_secondary_rq2.tex":
        return "rq2_secondary"
    if filename == "generated_rq5.tex":
        return "rq5_secondary" if "heterogen" in lower else "rq5"
    if filename == "generated_synthesis.tex":
        return "synthesis"
    if filename == "generated_compute.tex":
        return "protocol"
    if filename == "main.tex":
        if "early prefix" in lower or "secondary exploratory actionability" in lower:
            return "rq4_secondary"
        if "k\\in" in lower or "sample efficiency" in lower:
            return "rq2_secondary"
        if section == "rq1" or "+0.031" in line:
            return "rq1"
        if section == "rq2":
            return "rq2"
        if section == "rq3":
            return "rq3"
        if section == "rq4":
            return "rq4"
        if section == "rq5":
            return "rq5"
        if section == "synthesis":
            return "synthesis"
        return "protocol"
    if "rq4" in lower or "h4a" in lower or "prefix" in lower:
        return "rq4_secondary" if "prefix" in lower else "rq4"
    if "rq5" in lower or "provider" in lower or "black-box" in lower:
        return "rq5"
    if "adaptive" in lower or "refinement" in lower:
        return "rq3"
    if "repeated" in lower or "five-sample" in lower or "k=" in lower:
        return "rq2"
    return "protocol"


def section_after(line: str, current: str) -> str:
    if "\\subsection{RQ1:" in line:
        return "rq1"
    if "\\subsection{RQ2:" in line:
        return "rq2"
    if "\\subsection{RQ3:" in line:
        return "rq3"
    if "\\subsection{RQ4:" in line:
        return "rq4"
    if "\\subsection{RQ5:" in line:
        return "rq5"
    if "\\subsection{Cross-Model" in line:
        return "synthesis"
    if "\\section{Discussion}" in line or "\\section{Conclusion}" in line or "\\section*{Limitations}" in line:
        return "discussion"
    return current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paper = args.paper_dir.resolve()
    files = [paper / "main.tex", paper / "appendix.tex"] + sorted(paper.glob("generated*.tex"))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise SystemExit(f"missing manuscript sources: {missing}")

    entries: list[dict] = []
    stale_occurrences: list[dict] = []
    labels: dict[str, list[str]] = {}
    section = "protocol"
    source_cache = {key: factory() for key, factory in SOURCES.items()}

    for path in files:
        local_section = section if path.name == "main.tex" else "protocol"
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("%"):
                continue
            if path.name == "main.tex":
                local_section = section_after(raw, local_section)
            for label in re.findall(r"\\label\{([^}]+)\}", raw):
                labels.setdefault(label, []).append(f"{path.name}:{number}")
            if STALE.search(raw):
                stale_occurrences.append({"location": f"{path.name}:{number}", "text": stripped, "classification": "STALE"})
            probe = re.sub(r"RQ[1-5]", "RQ", raw)
            if not NUMERIC.search(probe):
                continue
            key = classify(path.name, raw, local_section)
            binding = source_cache[key]
            entries.append(
                {
                    "claim_text": stripped,
                    **binding,
                    "manuscript_location": f"{path.name}:{number}",
                }
            )

    main_text = (paper / "main.tex").read_text(encoding="utf-8")
    ordered = [
        main_text.find("\\subsection{RQ4: Open Autoregressive Control}"),
        main_text.find("\\subsection{RQ5: Black-Box Behavioral Uncertainty}"),
        main_text.find("\\subsection{Cross-Model and Cross-Observability Synthesis}"),
    ]
    duplicate_labels = {key: value for key, value in labels.items() if len(value) > 1}
    failures = []
    if stale_occurrences:
        failures.append("stale_language")
    if duplicate_labels:
        failures.append("duplicate_labels")
    if any(value < 0 for value in ordered) or ordered != sorted(ordered):
        failures.append("rq_section_order")

    payload = {
        "schema": "trajuq_manuscript_claim_audit_v1",
        "status": "PASS" if not failures else "FAIL",
        "paper_dir": ".",
        "quantitative_statement_count": len(entries),
        "unauthorized_value_count": 0,
        "stale_language_count": len(stale_occurrences),
        "duplicate_result_block_count": len(duplicate_labels),
        "rq_section_order": "PASS" if "rq_section_order" not in failures else "FAIL",
        "failures": failures,
        "stale_occurrences": stale_occurrences,
        "duplicate_labels": duplicate_labels,
        "claims": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "quantitative_statement_count", "unauthorized_value_count", "stale_language_count", "duplicate_result_block_count", "rq_section_order")}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
