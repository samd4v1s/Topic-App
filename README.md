# Team Topic Modeler

This repository is a vibe-coded prototype for exploring topic modeling over CSV data using BERTopic. It combines a Python backend with a lightweight Streamlit UI so you can upload documents, train a topic model, inspect topics, and save/load models locally.

This project is intentionally experimental: the goal is rapid iteration and proof-of-concept exploration, not production-grade deployment.

## What it does

- Upload a CSV file through a Streamlit app
- Select a text column to analyze
- Train a BERTopic-based topic model locally
- View topic information in a dataframe
- Save and reload trained models from the local saved_models directory

## Tech stack

- Python
- uv for dependency management
- BERTopic
- Sentence Transformers
- UMAP
- HDBSCAN
- scikit-learn
- Streamlit

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e ".[dev]"
```

## Running the app

Start the Streamlit interface:

```bash
make run
```

Or run it directly:

```bash
uv run streamlit run src/app.py
```

## Running tests

```bash
make test
```

## Project structure

- src/topic_modeler.py - backend topic modeling class
- src/app.py - Streamlit frontend
- tests/test_topic_modeler.py - backend tests

## Notes

This is a prototype built quickly for exploration and demos. Expect rough edges, experimental behavior, and plenty of room for improvement.
