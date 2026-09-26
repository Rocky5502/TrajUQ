# Anonymous TrajUQ software supplement

This archive contains the implementation and reproducibility material for the
anonymous submission **When Diffusion Language Models Are Unsure:
Trajectory-Aware Uncertainty for Selective and Adaptive Generation**.

## Scope

Included:

- trajectory telemetry and feature extraction;
- model adapters and parity utilities;
- benchmark preparation manifests and deterministic split code;
- leak-safe scoring and run audits;
- calibration, selective prediction, paired statistics, ablations, macro
  aggregation, and adaptive-compute evaluation;
- synthetic smoke data and unit tests;
- reproducibility and model-specific instrumentation documentation.

Not included:

- author identities or affiliations;
- the public development-repository URL;
- paper source;
- raw benchmark caches;
- gated benchmark content;
- model weights;
- experiment outputs;
- local machine paths.

## Installation

Python 3.10+:

    pip install -e ".[dev,math]"

CPU verification:

    python -m compileall -q src scripts tests
    python -m ruff check src tests scripts
    python -m pytest -q
    python scripts/smoke_metrics.py
    python scripts/smoke_end_to_end.py --output-dir results/smoke_e2e

GPU model experiments additionally require:

    pip install -e ".[models]"

## Reproducibility contract

Primary result generation requires immutable model and dataset revisions,
model-specific parity gates, outcome-isolated raw generation, separate scoring,
and a frozen calibration/test protocol. See the documentation directory.

The included synthetic data are only for software tests. Benchmark data are
downloaded from their upstream sources subject to their original licenses and
access conditions.

## Artifact manifest

MANIFEST.json records SHA-256 hashes and sizes for the files in this archive.
