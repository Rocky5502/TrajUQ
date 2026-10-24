# Security policy

TrajUQ is research software. The highest-risk repository failures are accidental disclosure of credentials, gated data, private connection material, or provider authorization metadata.

## Never commit

- API keys, bearer tokens, SSH private keys, or passphrases
- environment files containing secrets
- rented-machine credential files
- cloud/provider credentials or model-access tokens
- raw gated benchmark rows
- model weights or model/dataset caches
- private identifiers that are not required for reproducibility

The repository .gitignore blocks common credential patterns. CI additionally runs scripts/security_scan.py over tracked files.

## Local secret handling

Keep operator secrets outside the repository. Pass credentials through environment variables or operator-owned secret files excluded by Git. Never print authentication headers, API keys, SSH keys, or secret file contents into logs.

RQ5 records provider-visible scientific metadata but excludes authorization headers and credentials from persisted artifacts.

## Dependency security

- Python and GitHub Actions dependencies are monitored through Dependabot.
- Use isolated virtual environments.
- Re-check upstream model/dataset licenses and trust-remote-code requirements before execution.
- Do not auto-upgrade the authoritative experiment environment.

## Reporting a vulnerability

Do not publish a credential, gated-data sample, or sensitive execution detail in a public issue. Use GitHub private security reporting/advisories when available, or contact the maintainer through a private channel.

## Scientific integrity is also a safety boundary

Security fixes must not silently change frozen experiment semantics. If a security repair changes model code, data preparation, decoding, scoring, or statistics, treat it as a new implementation revision and rerun the required scientific gates.
