.PHONY: install test test-critical lint typecheck format clean parity

install:
	uv pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

# The look-ahead test alone — run this before every commit
test-critical:
	pytest tests/test_look_ahead.py -v

# Gate 0 parity check (synthetic data, no vectorbt required for basic run)
parity:
	python3 scripts/parity_check.py --synthetic

parity-vbt:
	python3 scripts/parity_check.py --synthetic  # requires: pip install vectorbt

lint:
	ruff check mft/ tests/ scripts/ learn/

# Best-effort static typing pass — not part of the enforced lint standard
typecheck:
	mypy mft/ --ignore-missing-imports

format:
	ruff format mft/ tests/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache dist/ build/ *.egg-info/
