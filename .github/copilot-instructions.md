# Copilot instructions for word_forge

## Project context
- Python package under `src/word_forge` with a `word_forge` CLI entry point.
- Core dependencies listed in `pyproject.toml` include heavy ML/vector libs (torch, transformers, sentence-transformers, chromadb, faiss, plotly, pyvis).
- CI installs only a lightweight subset and tests often stub these heavy imports, so avoid relying on heavyweight functionality unless explicitly required.
- Demo/database scripts may emit SQLite files (e.g., `test_database.sqlite`, `db_worker_demo/`); treat them as disposable artifacts.

## Environment setup
- Use Python 3.10 in CI; project supports 3.8+.
- Minimal dev install (matches CI): `pip install networkx numpy black ruff pytest`. Full tooling via `pip install -e .[dev]` (adds isort, mypy, pytest-cov, pre-commit).
- Avoid installing `vector`/`visualization` extras unless the task truly needs real embeddings or graph rendering.

## Linting and tests
- Formatting: `black --check .` (line length 88).
- Linting: `ruff check . --exit-zero` (mirrors CI; drop `--exit-zero` locally to fail on lint errors).
- Tests: `pytest -q` (uses `tests/`).
- Current baseline (Jan 2026): `black --check` reports formatting changes for several modules, and `pytest` fails at least:
  - `tests/test_graph_manager.py::test_graph_includes_emotional_relationships` (EmotionManager dummy signature).
  - `tests/test_vector_store.py::test_sqlite_faiss_fallback_used_when_chromadb_missing` (faiss dependency missing).
  Note these before attributing failures to new changes.

## Working guidelines
- Respect existing configuration in `pyproject.toml` (Black, Ruff, mypy settings).
- When touching vector/graph/emotion code, keep optional dependencies guarded and compatible with the lightweight CI environment.
- Do not commit generated datasets, model files, or demo SQLite outputs.
