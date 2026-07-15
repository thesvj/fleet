.PHONY: test lint format check install sim

export PYTHONPATH := $(CURDIR)

install:
	pip install -e ".[dev]"

lint:
	ruff check fleet tests examples
	ruff format --check fleet tests examples

format:
	ruff format fleet tests examples
	ruff check --fix fleet tests examples

# Offline-safe suite (stdlib). Prefer this when PyPI is unreachable.
test:
	python tests/run_sim.py
	python tests/run_failures.py

sim-only:
	python tests/run_sim.py

failures:
	python tests/run_failures.py

# Full pytest when dev deps installed
pytest:
	python -m pytest tests/ -q

check: test
	@command -v ruff >/dev/null && ruff check fleet tests examples || echo "ruff not installed — skip lint"

sim: test
	@rm -rf /tmp/fleet-sim-make
	@FLEET_HOME=/tmp/fleet-sim-make python -m fleet start -c mock -f short:2 --wait --timeout 30
	@FLEET_HOME=/tmp/fleet-sim-make python -m fleet run -c mock -- python examples/smoke_env.py
	@FLEET_HOME=/tmp/fleet-sim-make python -m fleet stop -c mock
	@echo "sim OK"
