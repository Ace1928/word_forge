# Repository Guidelines

## Project Structure & Module Organization
- `src/word_forge/` is the core package, organized by domain (`queue/`, `database/`, `parser/`, `vectorizer/`, `emotion/`, `graph/`, and `conversation/`).
- `tests/` contains unit and integration tests that exercise ingestion, graph/vector flows, and conversation pipelines.
- `data/samples/` contains only tiny, reviewable fixtures. Runtime data belongs
  in the platform data directory or an explicit `WORD_FORGE_HOME`.
- `docs/` contains user, architecture, security, performance, and upgrade guides.
- `webui/` is the editable TypeScript source for the packaged local workspace.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create and activate a local virtualenv.
- `python -m pip install -e ".[dev,gui]"`: install the development environment.
- `python -m pytest`: run the full test suite.
- `word_forge start hydrogen --minutes 0.1 --workers 2`: run a short crawl/ingest loop.
- `word_forge graph build --timeout 30`: build the graph from DB data.
- `word_forge vector index --embedder sentence-transformers/all-MiniLM-L6-v2`: build vector indices.

## Coding Style & Naming Conventions
- Python 3.10+ with type hints; follow the `mypy` settings in `pyproject.toml`.
- Format and lint with Ruff using the repository configuration.
- Use descriptive class names (e.g., `GraphManager`, `QueueWorker`) and snake_case for functions.

## Testing Guidelines
- Use `pytest`; name files `test_*.py` and keep tests close to the feature they exercise.
- Integration tests should validate real local behavior. External providers may
  be replaced with deterministic fakes; the default suite must not download models.
- When adding new features, include failure-mode coverage (missing data, queue state, DB errors).

## Commit & Pull Request Guidelines
- Commit messages are imperative and scoped (examples: `fix: start queue lifecycle`, `docs: update CLI flags`).
- PRs should include a short summary, test results, and any data/DB migration notes.
- If a change alters runtime artifacts (graphs, vectors), document the regeneration command.

## Local Data & Configuration Tips
- Prefer `--db-path` for isolated test runs; avoid overwriting shared DBs.
- Keep generated outputs in the platform runtime directories rather than the checkout.
