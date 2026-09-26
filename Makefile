.PHONY: test lint smoke env preflight

test:
	pytest -q

lint:
	ruff check src tests scripts

smoke:
	python scripts/smoke_metrics.py

env:
	python scripts/environment_snapshot.py --output results/environment.json

preflight:
	@echo "Usage: python scripts/preflight.py --model-config <yaml> --data <jsonl> --require-cuda"
