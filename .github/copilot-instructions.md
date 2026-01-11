# Copilot instructions for word_forge

## Project context
- Python package under `src/word_forge` with a `word_forge` CLI entry point.
- Optional vector/visualization features rely on heavy deps (torch, sentence-transformers, chromadb, faiss, plotly, pyvis); tests typically stub these and CI does not install them.
- Demo/database scripts may emit SQLite files (e.g., `test_database.sqlite`, `db_worker_demo/`); treat them as disposable artifacts.

## Environment setup
- Use Python 3.10 in CI; project supports 3.8+.
- Minimal dev install (matches CI): `pip install networkx numpy black ruff pytest` or `pip install -e .[dev]`.
- Avoid installing `vector`/`visualization` extras unless the task truly needs real embeddings or graph rendering.

## Linting and tests
- Formatting: `black --check .` (line length 88). Linting: `ruff check .` (exit-zero used in CI).
- Tests: `pytest -q` (uses `tests/`).
- Current baseline (Jan 2026): `black --check` reports formatting changes for several modules, and `pytest` fails at least:
  - `tests/test_graph_manager.py::test_graph_includes_emotional_relationships` (EmotionManager dummy signature).
  - `tests/test_vector_store.py::test_sqlite_faiss_fallback_used_when_chromadb_missing` (faiss dependency missing).
  Note these before attributing failures to new changes.

## Working guidelines
- Respect existing configuration in `pyproject.toml` (Black, Ruff, mypy settings).
- When touching vector/graph/emotion code, keep optional dependencies guarded and compatible with the lightweight CI environment.
- Do not commit generated datasets, model files, or demo SQLite outputs.
