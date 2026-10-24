.PHONY: help install-analysis test lint smoke security reproduce verify status env release preflight

help:
	@echo "TrajUQ reviewer/developer commands"
	@echo "  make reproduce       Validate released authorized evidence (CPU-safe)"
	@echo "  make verify          Security + lint + tests + metric smoke"
	@echo "  make status          Build the local project-status snapshot"
	@echo "  make release         Build deterministic code/reproducibility archive"
	@echo "  make env             Capture a local environment snapshot"

install-analysis:
	python -m pip install -r requirements-analysis.txt

test:
	pytest -q

lint:
	ruff check src tests scripts

smoke:
	python scripts/smoke_metrics.py

security:
	python scripts/security_scan.py --repo .

reproduce:
	python reproduction/reproduce_core.py

verify: security lint test smoke

status:
	python scripts/project_status.py

env:
	python scripts/environment_snapshot.py --output results/environment.json

release:
	python scripts/package_release.py

preflight:
	@echo "Usage: python scripts/preflight.py --model-config <yaml> --data <jsonl> --require-cuda"
