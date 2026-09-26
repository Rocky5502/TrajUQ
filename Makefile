.PHONY: install-dev test lint smoke audit-env

install-dev:
	python -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests scripts

smoke:
	python scripts/smoke_metrics.py

audit-env:
	python scripts/environment_snapshot.py
