#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve a mutable Hugging Face repo name/revision to an immutable commit SHA."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", choices=["model", "dataset"], required=True)
    parser.add_argument("--revision", default=None, help="Optional branch/tag/revision to resolve")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit('Install model extras first: pip install -e ".[models]"') from exc

    api = HfApi()
    if args.repo_type == "model":
        info = api.model_info(args.repo_id, revision=args.revision)
    else:
        info = api.dataset_info(args.repo_id, revision=args.revision)

    sha = getattr(info, "sha", None)
    if not sha:
        raise SystemExit("Hugging Face did not return an immutable commit SHA")

    payload = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "requested_revision": args.revision,
        "resolved_sha": sha,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
