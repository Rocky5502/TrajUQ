# Reproducing TrajUQ

This directory provides the lightweight verification path for the research software repository.

## Core verification

The default path is CPU-safe and does not download model weights or benchmark rows:

~~~bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-analysis.txt
python reproduction/reproduce_core.py
~~~

This checks repository secret hygiene and the released authorized artifact manifest and hashes.

## Full CPU verification

~~~bash
pip install -e ".[dev,math]"
python scripts/verify_cpu.py --output results/cpu_verification.json --logs-dir results/cpu_verification_logs
~~~

The full CPU suite runs compile checks, Ruff, pytest, metric smoke tests, end-to-end synthetic smoke tests, and the pinned MATH-500 scorer audit.

## Full experiment reproduction

Full model generation is intentionally separate because it requires substantial compute, model access, and benchmark access.

Start with docs/EXPERIMENT_PROTOCOL.md, docs/DATASETS.md, docs/MODELS.md, docs/DATA_SPLITS.md, and docs/A100_EXECUTION.md.

Raw benchmark rows, model weights, caches, credentials, and manuscript source are not redistributed by this repository.
