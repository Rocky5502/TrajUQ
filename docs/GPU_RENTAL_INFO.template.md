# Private GPU rental information template

This file is a template only. Do **not** commit real credentials, tokens, SSH
private keys, passwords, provider cookies, billing identifiers, or access URLs
that embed secrets.

Recommended operator-private location:

```text
private/GPU_RENTAL_INFO.private.md
```

The `private/` directory and `GPU_RENTAL_INFO.private.*` are gitignored.

## Planned authoritative TrajUQ rental profile

Provider: gpu.ai-galaxy (operator-selected)
GPU: 1 x NVIDIA A100-SXM4-80GB
CPU: 16 cores
Host RAM: 48 GB
System disk: 200 GB
Network: approximately 250 Mbps in the selected offer
Image: Ubuntu 24.04 with CUDA 13.0
Operational experiment envelope: 72 continuous hours
Autonomous health-monitor cadence: 45 minutes

The repository's machine gate remains authoritative. The actual rented machine
must pass `scripts/verify_a100_machine.py`; screenshots/provider labels are not
substitutes for the generated PASS artifact.

## Operator-only fields

Store values below only in the ignored private copy, never in this tracked
template:

```text
INSTANCE_ID=
HOST=
SSH_PORT=
SSH_USER=
SSH_IDENTITY_FILE=
PROVIDER_CONSOLE_NOTES=
PERSISTENT_DISK_NOTES=
OVERLEAF_REMOTE_NOTES=
```

## Security rules for Codex / automation

- Never print secrets in terminal summaries, paper files, Git commits, issues,
  handoff ZIPs, or experiment artifacts.
- Never copy private key contents into the repository.
- Prefer an SSH config alias or environment variable supplied by the operator.
- Hugging Face authentication is interactive/operator-owned. Use `hf auth
  whoami` to verify it without displaying a token.
- If an Overleaf Git remote is configured, credentials must remain in the
  operator's credential manager/SSH agent. Otherwise build the deterministic
  Overleaf ZIP instead of pretending a remote sync occurred.
