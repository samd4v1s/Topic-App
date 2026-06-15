.PHONY: install test run clean

install:
	uv venv
	uv pip install -e ".[dev]"

test:
	uv run pytest

run:
	uv run streamlit run src/app.py

clean:
	rm -rf .venv .pytest_cache __pycache__ ./saved_models
