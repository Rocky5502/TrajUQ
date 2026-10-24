# Reviewer guide

This is the shortest path through TrajUQ for a reviewer, artifact evaluator, or collaborator.

## 1. Verify the released evidence layer

No GPU, model weights, or benchmark rows are required.

~~~bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-analysis.txt
python reproduction/reproduce_core.py
~~~

Expected outcome:

~~~text
SECURITY_SCAN=PASS
...
TRAJUQ_CORE_REPRODUCTION=PASS
~~~

The command validates the tracked-file security scan and the SHA-256-bound authorized artifact manifest.

## 2. Understand the scientific design

Read these in order:

1. [RESEARCH_QUESTIONS.md](RESEARCH_QUESTIONS.md) — what RQ1–RQ5 ask and what each RQ is allowed to claim.
2. [MODELS.md](MODELS.md) — exact open-weight revisions, black-box provider identifiers, decoding settings, and telemetry contracts.
3. [DATASETS.md](DATASETS.md) — exact dataset sources, revisions, split policies, caps, and access boundaries.
4. [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) — the end-to-end evidence contract.
5. [EVIDENCE_MAP.md](EVIDENCE_MAP.md) — where registered evidence lives and which items are authorized versus pending.

## 3. Inspect the primary provenance manifests

Machine-readable reviewer entry points:

- \`manifests/models.json\`
- \`manifests/datasets.json\`
- \`manifests/environment.json\`
- \`ARTIFACT_MANIFEST.json\`
- \`configs/experiments/primary.yaml\`
- \`data/manifests/evaluation_splits.json\`

The original frozen experiment implementation is commit:

\`28002055c828c6910840de6f2105aa6367514144\`

## 4. Run deeper CPU checks

~~~bash
pip install -e ".[dev,math]"
make verify
~~~

This runs:
- repository security hygiene;
- Ruff;
- pytest;
- metric smoke tests.

For the complete archived CPU verification protocol, use:

~~~bash
python scripts/verify_cpu.py \
  --output results/cpu_verification.json \
  --logs-dir results/cpu_verification_logs
~~~

## 5. What is intentionally not in this repository?

TrajUQ does not redistribute:
- raw/gated benchmark rows;
- model weights;
- model/dataset caches;
- API credentials;
- private machine credentials;
- manuscript or venue-authoring source.

These omissions are deliberate reproducibility/security boundaries, not missing experimental definitions. Exact upstream identifiers and revisions are recorded instead.

## 6. What not to infer

- A generated file is not automatically an audited result.
- A progress percentage is not a scientific conclusion.
- A black-box provider model ID is not evidence of a fixed backend weight revision.
- A positive aggregate does not imply every model/dataset cell improved.
- RQ4 and RQ5 do not retroactively alter the original RQ1 diffusion-family gate.

## 7. Full GPU reproduction

Only reviewers who need full generation should proceed to [A100_EXECUTION.md](A100_EXECUTION.md). The GPU path is intentionally strict and fail-closed.
