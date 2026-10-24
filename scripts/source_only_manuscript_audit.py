#!/usr/bin/env python
"""Audit only canonical manuscript sources and their generated interfaces.

The audit deliberately excludes tests, build trees, temporary directories, and
release archives.  It distinguishes authorized execution-time placeholders
from submission blockers so operational progress can never masquerade as final
scientific evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCOPES = (
    Path("paper"),
    Path("results/generated"),
    Path("configs"),
    Path("scripts"),
    Path("src"),
    Path("CLAIM_EVIDENCE_REGISTRY.md"),
)
TEXT_SUFFIXES = {".bib", ".json", ".md", ".py", ".tex", ".yaml", ".yml"}
EXCLUDED_PARTS = {
    "__pycache__", "dist", "tmp", ".pytest_cache", ".ruff_cache",
}
MARKERS = re.compile(
    r"\b(?:TODO|TBD|PLACEHOLDER|FIXME|XXX|NOT YET AUTHORIZED)\b",
    re.IGNORECASE,
)
PROVISIONAL = re.compile(r"\b(?:pending|provisional)\b", re.IGNORECASE)
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:auto|page|eq)?ref\{([^}]+)\}")
CITE_RE = re.compile(r"\\cite[a-zA-Z*]*\{([^}]+)\}")
BIB_RE = re.compile(r"^\s*@\w+\s*\{\s*([^,\s]+)", re.MULTILINE)
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
MACRO_DEF_RE = re.compile(r"\\(?:newcommand|renewcommand|providecommand)\s*\{?\\([A-Za-z@]+)")
MACRO_USE_RE = re.compile(r"\\([A-Z][A-Za-z@]+)\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for scope in SCOPES:
        candidate = root / scope
        if candidate.is_file():
            files.append(candidate)
            continue
        if not candidate.exists():
            continue
        for path in candidate.rglob("*"):
            relative = path.relative_to(root)
            if (
                path.is_file()
                and path.suffix.lower() in TEXT_SUFFIXES
                and not any(part in EXCLUDED_PARTS or part.startswith("build-") for part in relative.parts)
            ):
                files.append(path)
    return sorted(set(files))


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def resolve_graphic(tex: Path, value: str) -> Path | None:
    base = tex.parent / value
    candidates = [base] if base.suffix else [base.with_suffix(ext) for ext in (".pdf", ".png", ".svg", ".eps")]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def audit(root: Path) -> dict[str, object]:
    files = source_files(root)
    texts = {path: path.read_text(encoding="utf-8", errors="replace") for path in files}
    tex_files = [path for path in files if path.suffix.lower() == ".tex"]

    markers: list[dict[str, object]] = []
    provisional: list[dict[str, object]] = []
    for path, text in texts.items():
        for line_no, line in enumerate(text.splitlines(), 1):
            for match in MARKERS.finditer(line):
                markers.append({"path": rel(path, root), "line": line_no, "token": match.group(0)})
            for match in PROVISIONAL.finditer(line):
                provisional.append({"path": rel(path, root), "line": line_no, "token": match.group(0)})

    labels: dict[str, list[str]] = defaultdict(list)
    refs: dict[str, list[str]] = defaultdict(list)
    citations: dict[str, list[str]] = defaultdict(list)
    missing_figures: list[dict[str, str]] = []
    for path in tex_files:
        text = texts[path]
        for name in LABEL_RE.findall(text):
            labels[name].append(rel(path, root))
        for name in REF_RE.findall(text):
            refs[name].append(rel(path, root))
        for group in CITE_RE.findall(text):
            for key in group.split(","):
                citations[key.strip()].append(rel(path, root))
        for match in GRAPHICS_RE.finditer(text):
            value = match.group(1)
            if resolve_graphic(path, value) is None:
                prefix = text[max(0, match.start() - 180):match.start()]
                conditional = f"\\IfFileExists{{{value}}}" in prefix
                missing_figures.append({
                    "source": rel(path, root), "target": value,
                    "status": "DEFERRED_CONDITIONAL" if conditional else "MISSING_REQUIRED",
                })

    bib_keys: set[str] = set()
    for path, text in texts.items():
        if path.suffix.lower() == ".bib":
            bib_keys.update(BIB_RE.findall(text))

    definitions: set[str] = set()
    uses: dict[str, list[str]] = defaultdict(list)
    for path in tex_files:
        definitions.update(MACRO_DEF_RE.findall(texts[path]))
        for name in MACRO_USE_RE.findall(texts[path]):
            uses[name].append(rel(path, root))
    # Project macros are intentionally PascalCase; common LaTeX commands are
    # lower-case and therefore excluded by construction.
    standard_uppercase = {"Delta", "IfFileExists", "Vert"}
    undefined_project_macros = sorted(
        name for name in uses if name not in definitions and name not in standard_uppercase
    )

    marker_counts = Counter(item["token"].upper() for item in markers)
    intentional_placeholders = []
    for item in markers:
        path = str(item["path"])
        line = int(item["line"])
        token = str(item["token"]).upper()
        intentional = (
            path.startswith("paper/generated")
            or path == "CLAIM_EVIDENCE_REGISTRY.md"
            or path.startswith("configs/templates/")
            or path.startswith("scripts/")
            or path == "paper/README.md"
            or (
                path == "paper/main.tex"
                and token == "TBD"
                and r"\newcommand{\tbd}{\textsc{TBD}}" in texts[root / Path(path)]
            )
        )
        if intentional:
            intentional_placeholders.append(item)
    blocker_markers = [item for item in markers if item not in intentional_placeholders]
    duplicate_labels = {key: value for key, value in sorted(labels.items()) if len(value) > 1}
    undefined_refs = {key: value for key, value in sorted(refs.items()) if key not in labels}
    missing_citations = {key: value for key, value in sorted(citations.items()) if key not in bib_keys}

    blockers: list[str] = []
    if blocker_markers:
        blockers.append(f"{len(blocker_markers)} source placeholder marker(s) outside generated interfaces/registry")
    if duplicate_labels:
        blockers.append(f"{len(duplicate_labels)} duplicate LaTeX label(s)")
    if undefined_refs:
        blockers.append(f"{len(undefined_refs)} undefined LaTeX reference(s)")
    if missing_citations:
        blockers.append(f"{len(missing_citations)} missing bibliography key(s)")
    required_missing_figures = [item for item in missing_figures if item["status"] == "MISSING_REQUIRED"]
    if required_missing_figures:
        blockers.append(f"{len(required_missing_figures)} missing required figure asset(s)")
    if undefined_project_macros:
        blockers.append(f"{len(undefined_project_macros)} undefined project macro(s)")

    file_bindings = [
        {"path": rel(path, root), "sha256": sha256_file(path)} for path in files
    ]
    return {
        "schema": "trajuq_source_only_manuscript_audit_v1",
        "scope": [scope.as_posix() for scope in SCOPES],
        "excluded": sorted(EXCLUDED_PARTS),
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "counts": {
            "source_files": len(files),
            "placeholder_markers": len(markers),
            "intentional_placeholders": len(intentional_placeholders),
            "blocking_placeholders": len(blocker_markers),
            "provisional_terms": len(provisional),
            "labels": len(labels),
            "references": len(refs),
            "citations": len(citations),
            "bibliography_entries": len(bib_keys),
        },
        "marker_counts": dict(sorted(marker_counts.items())),
        "blocking_placeholder_locations": blocker_markers,
        "duplicate_labels": duplicate_labels,
        "undefined_references": undefined_refs,
        "missing_citations": missing_citations,
        "missing_figures": missing_figures,
        "undefined_project_macros": undefined_project_macros,
        "files": file_bindings,
        "evidence_firewall": {
            "generated_placeholders_are_authorized_for_submission": False,
            "progress_counts_are_scientific_evidence": False,
            "pending_values_are_estimates": False,
        },
    }


def markdown(report: dict[str, object]) -> str:
    counts = report["counts"]
    assert isinstance(counts, dict)
    lines = [
        "# Source-only manuscript audit",
        "",
        f"**Status:** `{report['status']}`",
        "",
        "This audit scans only canonical manuscript/code/config sources. Tests, temporary trees, build outputs, pytest sandboxes, and release archives are excluded.",
        "",
        "## Evidence firewall",
        "",
        "- Generated `TBD`, `pending`, and `NOT YET AUTHORIZED` entries remain execution-time interfaces, not scientific results.",
        "- Operational progress counters cannot authorize manuscript claims.",
        "- Missing empirical values are never estimated or copied by hand.",
        "",
        "## Counts",
        "",
        "| Check | Count |",
        "|---|---:|",
    ]
    for key in (
        "source_files", "placeholder_markers", "intentional_placeholders",
        "blocking_placeholders", "provisional_terms", "labels", "references",
        "citations", "bibliography_entries",
    ):
        lines.append(f"| {key.replace('_', ' ')} | {counts[key]} |")
    lines.extend(["", "## Submission blockers", ""])
    blockers = report["blockers"]
    assert isinstance(blockers, list)
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- None in the source-only structural audit.")
    lines.extend([
        "",
        "## Intentional unfinished evidence",
        "",
        "The current paper remains pre-final wherever RQ2--RQ5 or cross-regime synthesis gates are incomplete. These placeholders are allowed only because the prose labels them as pending/provisional and the final submission gate rejects them.",
        "",
        "## Machine-readable companion",
        "",
        "See `SOURCE_ONLY_MANUSCRIPT_AUDIT.json` for path-level findings and SHA-256 bindings.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output", default="SOURCE_ONLY_MANUSCRIPT_AUDIT.md")
    parser.add_argument("--json-output", default="SOURCE_ONLY_MANUSCRIPT_AUDIT.json")
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    report = audit(root)
    (root / args.output).write_text(markdown(report), encoding="utf-8", newline="\n")
    (root / args.json_output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"status": report["status"], "blockers": report["blockers"]}))
    return 2 if args.require_clean and report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
