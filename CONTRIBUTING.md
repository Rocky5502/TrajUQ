# Contributing to TrajUQ

Contributions are welcome when they improve correctness, reproducibility, documentation, testing, or clearly separated research extensions.

## Before opening a change

1. Create an isolated branch.
2. Keep secrets, raw gated data, model weights, caches, and manuscript source out of the repository.
3. Preserve the distinction between the authoritative experiment commit and later repository-engineering commits.
4. Do not modify frozen scientific settings simply to improve an observed result.
5. Add or update tests for behavior changes.

## Local checks

~~~bash
pip install -e ".[dev,math]"
python scripts/security_scan.py --repo .
ruff check src scripts tests
pytest -q
~~~

For experiment changes, also run the relevant parity, provenance, freeze, and audit gates described in docs/EXPERIMENT_PROTOCOL.md.

## Pull requests

A good pull request explains what changed, why it changed, whether scientific semantics changed, what tests were run, and whether any model/dataset/config revision, reproducibility contract, or license boundary changed.

## Research extensions

New experiments should use separate prospective configs rather than rewriting completed confirmatory protocols. Negative or null outcomes must not be selectively removed.

## Documentation

Prefer durable scientific documentation over chat transcripts or operator prompts. Internal manuscript/editorial files are intentionally maintained outside this repository.
