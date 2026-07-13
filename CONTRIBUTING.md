# Contributing to Word Forge

Thank you for improving Word Forge. Contributions should make the product more
correct, understandable, portable, or useful without silently increasing the
core installation burden.

## Set up

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

## Before opening a pull request

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy --follow-imports=skip \
  src/word_forge/{lexicon,models,network,paths,services,observability,cli,_version}.py \
  src/word_forge/web
python -m pytest
npm ci
npm test
npm run build
python -m build
python -m twine check dist/*
```

Add tests for behavior and failure modes. Update user documentation when a
command, public API, data model, dependency, environment variable, or runtime
path changes.

## Design rules

- Keep core imports lightweight and offline-capable.
- Guard optional dependencies and give the exact install command when missing.
- Parameterize SQL and validate untrusted input at public boundaries.
- Keep generated data, model files, databases, reports, and profiles out of Git.
- Prefer typed public models and small single-purpose functions.
- Preserve Unicode data; keep new source-code messages and documentation
  punctuation ASCII where practical for terminal compatibility.
- Do not make network access or model downloads implicit.

## Commits and pull requests

Use an imperative, scoped subject such as `feat(cli): add portable export` or
`fix(database): roll back failed lexeme batch`. A pull request should explain:

1. The user-visible problem.
2. The chosen behavior and important tradeoffs.
3. Tests and platforms used.
4. Data migration or regeneration requirements.
5. Security, privacy, and performance effects.

By participating, you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
