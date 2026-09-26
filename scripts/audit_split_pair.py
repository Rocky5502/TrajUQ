#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajuq.data_integrity import audit_split_pair


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a frozen calibration/test pair for leakage and manifest drift."
    )
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    payload = audit_split_pair(
        args.calibration,
        args.test,
        args.manifest,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)

    output = (
        Path(args.output)
        if args.output
        else Path(args.manifest).with_name("split_audit.json")
    )
    output.write_text(rendered + "\n", encoding="utf-8")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
