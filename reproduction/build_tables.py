#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    required = [
        "paper/generated_rq1.tex",
        "paper/generated_authorized_rq23.tex",
        "paper/generated_secondary_rq2.tex",
        "paper/generated_rq5.tex",
        "paper/generated_synthesis.tex",
    ]
    failures = [path for path in required if not (ROOT / path).is_file()]
    rq5 = (ROOT / "paper/generated_rq5.tex").read_text(encoding="utf-8")
    for token in ("-0.0030", "0.6877", "not supported", "non-identifiable"):
        if token.lower() not in rq5.lower():
            failures.append(f"paper/generated_rq5.tex missing {token!r}")
    print(json.dumps({"status": "PASS" if not failures else "FAIL", "checked": required, "failures": failures}, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
