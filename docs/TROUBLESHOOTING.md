# Troubleshooting

## Core reproduction fails on a missing or changed artifact

Run:

~~~bash
python scripts/validate_release_manifest.py
~~~

A hash mismatch means the tracked authorized artifact differs from the manifest. Do not update the expected hash casually. First determine whether the artifact was intentionally regenerated under an authorized protocol.

## Security scan fails

Run:

~~~bash
python scripts/security_scan.py --repo .
~~~

Do not whitelist real credentials. Remove the credential from tracked files and rotate it if it was ever valid.

## GitHub Actions fails before any step starts

If an Actions job shows no allocated steps/logs, treat it as runner/infrastructure failure rather than a scientific-test failure. Re-run the job or run the repository checks locally:

~~~bash
make verify
~~~

A real code/test failure has an executed step and diagnostic output.

## GPQA cannot be downloaded

GPQA-Diamond is gated. Obtain access under the upstream dataset terms. TrajUQ does not include or bypass gated benchmark rows.

## Hugging Face model access fails

Confirm:
- you are authenticated locally;
- your account can access the requested checkpoint;
- the requested revision matches the frozen manifest;
- no moving branch/head is being substituted for the pinned SHA.

Never paste access tokens into logs or repository files.

## A model parity gate fails

Stop. A parity mismatch is a scientific stop condition.

Do not:
- lower the parity requirement;
- switch checkpoints;
- alter seeds/settings after seeing outcomes;
- keep incompatible telemetry as confirmatory evidence.

Repair the implementation in a new commit and repeat the required freeze/parity chain.

## Frozen revision drift

Check:
- \`manifests/models.json\`;
- \`manifests/datasets.json\`;
- \`results/revision_ledger.json\` when reproducing a full frozen run.

The registered exact revision wins over a moving upstream default branch.

## MATH-500 scorer problems

Use the pinned symbolic-equivalence environment described in [MATH500_SCORING.md](MATH500_SCORING.md). Do not silently replace it with string equality.

## Black-box API execution

RQ5 credentials must be supplied outside Git. Persist provider-visible scientific metadata, not authorization headers or secret values.

Provider-side revisions, routing, hardware, and precision are not assumed observable from the API identifier.

## CUDA / GPU mismatch

The authoritative open-weight campaign environment is summarized in \`manifests/environment.json\`. Full reproduction should match the registered device/dtype contract or be reported as a new environment rather than silently mixed with authoritative artifacts.
