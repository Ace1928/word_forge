# Word Forge

Word Forge is a lexical data processing and enrichment system designed to analyze and transform textual datasets. The project focuses on modular and documented components that follow functional design principles.

## License

This project is licensed under the terms of the [MIT License](LICENSE).
=======
Word Forge is a modular system for building and enriching a lexical database. It integrates multiple resources, including WordNet, OpenThesaurus and transformer-based models, to collect definitions, examples and semantic relations.

## NLTK Data

Several components rely on datasets distributed with NLTK. These files are downloaded automatically the first time Word Forge accesses WordNet or related features via `ensure_nltk_data()`.

Ensure the running environment has internet access on the initial run so these resources can be retrieved.

Word Forge is a modular lexical processing and enrichment toolkit. It builds a comprehensive lexical network while providing vector search, emotion analysis, and graph capabilities. The project embraces the "Eidosian" design philosophy—typed interfaces, clear separation of concerns, and recursive self‑improvement.

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies using the project metadata:
   ```bash
   pip install -e .
   ```
   Development tools (formatter, linter, tests) are available via:
   ```bash
   pip install -e .[dev]
   ```
   A `requirements.txt` file is also provided for environments that require it:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start Examples

- **Inspect configuration**
  ```bash
  python -m word_forge.demos.config_demo --validate
  ```
- **Run the vector worker demo**
  ```bash
  python -m word_forge.demos.vector_worker_demo
  ```
- **Generate lexical data for a word**
  ```bash
  python lexical_proto.py recursion
  ```

## Dependency Management

Project dependencies are declared in `pyproject.toml`. Optional development tools are under the `[project.optional-dependencies]` section. The `requirements.txt` file mirrors these packages for compatibility with tooling that does not yet read `pyproject.toml`.

## Running Tests

Install the development dependencies as shown above and execute:

```bash
pytest
```

Tests live in the `tests/` directory and use `pytest`. Configuration for pytest is stored in `pyproject.toml`.

