# Copilot instructions for word_forge

## Goals
- Keep repository guidance accurate for agents working on Word Forge (Python package under `src/word_forge` with a `word_forge` CLI entry point).
- Ensure dependency setup instructions cover both full installs (`pip install -e .[dev]`) and the lightweight CI footprint.
- Keep linting, testing, and known-baseline status current to avoid misattributing failures.
- After making changes, rerun relevant checks, verify results, and update this Goals/Current Status/TODO section accordingly.

## Current Status
- Python 3.10 in CI; project supports 3.8+.
- Full dependency install for local work: `pip install -e .[dev]` (installs declared runtime deps plus isort, mypy, pytest-cov, pre-commit, etc.). Core install alone pulls heavy ML/vector libs (torch, transformers, sentence-transformers, chromadb, faiss, plotly, pyvis).
- CI installs a lightweight subset only (`networkx`, `numpy`, `black`, `ruff`, `pytest`); tests often stub heavy imports.
- Baseline checks (Jan 2026):
  - `black --check .` reports formatting changes for several modules.
  - `pytest -q` fails at least:
    - `tests/test_graph_manager.py::test_graph_includes_emotional_relationships` (EmotionManager dummy signature).
    - `tests/test_vector_store.py::test_sqlite_faiss_fallback_used_when_chromadb_missing` (faiss dependency missing).
- Demo/database scripts may emit SQLite files (e.g., `test_database.sqlite`, `db_worker_demo/`); treat them as disposable artifacts.

## TODO (keep updated)
- When you finish a task, update Current Status and this TODO with any new failing checks, added tools, or dependency/command changes.
- Note any new baseline failures, added skip conditions, or extra setup steps required for new features.
- Remove resolved items promptly to keep this list actionable.

## Project context
- Python package under `src/word_forge` with a `word_forge` CLI entry point.
- Core dependencies listed in `pyproject.toml` include heavy ML/vector libs (torch, transformers, sentence-transformers, chromadb, faiss, plotly, pyvis).
- CI installs only a lightweight subset and tests often stub these heavy imports, so avoid relying on heavyweight functionality unless explicitly required.
- Demo/database scripts may emit SQLite files (e.g., `test_database.sqlite`, `db_worker_demo/`); treat them as disposable artifacts.

## Environment setup
- Prefer full local setup: `pip install -e .[dev]` to ensure all declared dependencies and dev tools are available (expect heavy ML downloads).
- Minimal CI-like setup: `pip install networkx numpy black ruff pytest` (used by GitHub Actions) — only for reproducing CI behavior.
- Optional extras:
  - Vector stack: `pip install -e .[vector]` (sentence-transformers, chromadb, faiss, pyyaml).
  - Visualization stack: `pip install -e .[visualization]` (pyvis, plotly).

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
