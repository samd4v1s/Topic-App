.PHONY: install test clean

install:
	uv venv
	uv pip install -e ".[dev]"

test:
	uv run pytest

clean:
	rm -rf .venv .pytest_cache __pycache__ ./saved_models
