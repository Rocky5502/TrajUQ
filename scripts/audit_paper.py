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

    failures: list[str] = []
    warnings: list[str] = []
    for path in (main_path, appendix_path, bib_path):
        if not path.exists():
            failures.append(f"missing required manuscript file: {path}")

    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2))
        raise SystemExit(1)

    main = main_path.read_text(encoding="utf-8")
    appendix = appendix_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")
    all_text = "\n".join([main, appendix, bib])

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

    tbd_count = len(re.findall(r"\\tbd\b|\bTBD\b", all_text))
    if tbd_count:
        message = f"development placeholders remain: {tbd_count}"
        if args.submission_ready:
            failures.append(message)
        else:
            warnings.append(message)

    if "Empirical results are intentionally omitted" not in main and tbd_count:
        warnings.append("TBD values exist without explicit pre-results disclosure")

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "submission_ready_mode": bool(args.submission_ready),
        "tbd_count": tbd_count,
        "warnings": warnings,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
