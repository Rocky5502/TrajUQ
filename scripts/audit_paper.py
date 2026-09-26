#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static anonymity/format guard for the TrajUQ ARR manuscript."
    )
    parser.add_argument("--paper-dir", default="paper")
    parser.add_argument(
        "--submission-ready",
        action="store_true",
        help="Also fail on development TBD placeholders.",
    )
    args = parser.parse_args()

    root = Path(args.paper_dir)
    main_path = root / "main.tex"
    appendix_path = root / "appendix.tex"
    bib_path = root / "references.bib"
    generated_path = root / "generated_results.tex"

    failures: list[str] = []
    warnings: list[str] = []
    for path in (main_path, appendix_path, bib_path, generated_path):
        if not path.exists():
            failures.append(f"missing required manuscript file: {path}")

    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2))
        raise SystemExit(1)

    main = main_path.read_text(encoding="utf-8")
    appendix = appendix_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")
    generated = generated_path.read_text(encoding="utf-8")
    all_text = "\n".join([main, appendix, bib, generated])

    if r"\usepackage[review]{acl}" not in main:
        failures.append("main.tex must use the official ACL style in review mode")
    if r"\author{Anonymous ACL Submission}" not in main:
        failures.append("anonymous review author line is missing or changed")

    required_sections = [
        r"\section{Conclusion}",
        r"\section*{Limitations}",
    ]
    for section in required_sections:
        if section not in main:
            failures.append(f"missing required section: {section}")

    conclusion_pos = main.find(r"\section{Conclusion}")
    limitations_pos = main.find(r"\section*{Limitations}")
    if conclusion_pos >= 0 and limitations_pos >= 0 and limitations_pos < conclusion_pos:
        failures.append("Limitations must appear after the conclusion")

    if r"\bibliography{references}" not in main:
        failures.append("main.tex does not include references.bib")
    if r"\input{appendix}" not in main:
        failures.append("main.tex does not include appendix.tex in the same PDF")

    banned_patterns = {
        "public GitHub URL": r"https?://(?:www\.)?github\.com/",
        "tracking/storage URL": r"https?://(?:www\.)?(?:dropbox|drive\.google)\.",
        "Windows absolute path": r"[A-Za-z]:\\",
        "Unix home path": r"/home/[^/\s]+/",
        "acknowledgements heading": r"\\section\*?\{Acknowledg(?:e)?ments?\}",
    }
    for label, pattern in banned_patterns.items():
        if re.search(pattern, all_text, flags=re.IGNORECASE):
            failures.append(f"anonymous manuscript contains {label}")

    if re.search(r"\\author\{(?!Anonymous ACL Submission\})", main):
        failures.append("non-anonymous author declaration detected")

    citation_keys: set[str] = set()
    for match in re.finditer(
        r"\\cite(?:t|p|alp|author|year|yearpar)?\*?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^}]+)\}",
        main + "\n" + appendix,
    ):
        citation_keys.update(
            key.strip()
            for key in match.group(1).split(",")
            if key.strip()
        )

    bib_keys = re.findall(
        r"(?m)^\s*@\w+\s*\{\s*([^,\s]+)\s*,",
        bib,
    )
    duplicate_bib_keys = sorted(
        key for key in set(bib_keys) if bib_keys.count(key) > 1
    )
    if duplicate_bib_keys:
        failures.append(
            f"duplicate bibliography keys: {duplicate_bib_keys}"
        )

    bib_key_set = set(bib_keys)
    missing_citations = sorted(citation_keys - bib_key_set)
    if missing_citations:
        failures.append(
            f"citation keys missing from references.bib: {missing_citations}"
        )
    unused_bib = sorted(bib_key_set - citation_keys)
    if unused_bib:
        warnings.append(
            f"unused bibliography entries: {unused_bib}"
        )

    labels = re.findall(r"\\label\{([^}]+)\}", main + "\n" + appendix)
    duplicate_labels = sorted(
        label for label in set(labels) if labels.count(label) > 1
    )
    if duplicate_labels:
        failures.append(f"duplicate LaTeX labels: {duplicate_labels}")

    refs = set(
        re.findall(
            r"\\(?:ref|autoref|pageref)\{([^}]+)\}",
            main + "\n" + appendix,
        )
    )
    missing_labels = sorted(refs - set(labels))
    if missing_labels:
        failures.append(f"LaTeX references to missing labels: {missing_labels}")

    tbd_count = len(re.findall(r"\\tbd\b|\bTBD\b", all_text))
    if tbd_count:
        message = f"development placeholders remain: {tbd_count}"
        if args.submission_ready:
            failures.append(message)
        else:
            warnings.append(message)

    if (
        "Empirical results are intentionally omitted" not in all_text
        and "PRE-RESULTS FALLBACK" not in generated
        and tbd_count
    ):
        warnings.append("TBD values exist without explicit pre-results disclosure")

    if args.submission_ready:
        pre_results_markers = [
            "PRE-RESULTS FALLBACK",
            "pre-experiment manuscript",
            "This manuscript is currently pre-experiment",
        ]
        remaining = [
            marker
            for marker in pre_results_markers
            if marker in all_text
        ]
        if remaining:
            failures.append(
                f"pre-results language remains in submission-ready manuscript: {remaining}"
            )

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "submission_ready_mode": bool(args.submission_ready),
        "tbd_count": tbd_count,
        "citation_count": len(citation_keys),
        "bibliography_entry_count": len(bib_keys),
        "warnings": warnings,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
