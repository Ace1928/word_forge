# Word Forge

[![CI](https://github.com/Ace1928/word_forge/actions/workflows/ci.yml/badge.svg)](https://github.com/Ace1928/word_forge/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Word Forge is a modular lexical processing and enrichment toolkit that builds a comprehensive semantic network while providing vector search, emotion analysis, and graph visualization capabilities. The project embraces the **Eidosian** design philosophy—typed interfaces, clear separation of concerns, and recursive self‑improvement.

## Features

- **Lexical Data Aggregation**: Combines data from WordNet, thesauruses, and other linguistic resources
- **Semantic Graph**: Builds a multidimensional knowledge graph using NetworkX with relationship types including synonyms, antonyms, hypernyms, and emotional associations
- **Emotion Analysis**: Dimensional (valence/arousal) and categorical emotion detection using VADER, TextBlob, and optional LLM integration
- **Vector Search**: Semantic similarity search powered by sentence transformers and ChromaDB/FAISS backends
- **CLI Interface**: Full-featured command-line interface for all major operations
- **Background Workers**: Threaded workers for graph building, vector indexing, and emotion annotation

## Architecture

```
┌────────────────────┐
│    word_forge      │  CLI entry point
└─────────┬──────────┘
          │
┌─────────▼──────────┐
│       config       │  Centralized configuration
└─────────┬──────────┘
          │
┌─────────┴─────────────────────────────────────────┐
│                    Core Modules                    │
├──────────┬───────────┬──────────┬────────────────┤
│ database │   graph   │ emotion  │   vectorizer   │
│(SQLite)  │(NetworkX) │(VADER/TB)│ (Transformers) │
└──────────┴───────────┴──────────┴────────────────┘
          │
┌─────────▼──────────┐
│   queue/workers    │  Background processing
└────────────────────┘
```

## Installation

Word Forge targets **Python 3.8 or newer**.

### Basic Installation

```bash
# Clone and install locally
git clone https://github.com/Ace1928/word_forge.git
cd word_forge
pip install -e .

# Or install directly from Git
pip install git+https://github.com/Ace1928/word_forge.git
```

### Development Installation

Install with development tools (formatter, linter, tests):

```bash
pip install -e .[dev]
```

### Optional Feature Extras

Install feature bundles based on your needs:

| Extra | Command | Includes |
|-------|---------|----------|
| `vector` | `pip install -e .[vector]` | sentence-transformers, ChromaDB, FAISS |
| `visualization` | `pip install -e .[visualization]` | Pyvis, Plotly |
| `dev` | `pip install -e .[dev]` | black, ruff, pytest, mypy, pre-commit |

**Note**: The `word_forge` CLI relies on the `vector` extra for semantic search operations.

## Quick Start

### Python API

```python
from word_forge.config import config
from word_forge.database.database_manager import DBManager

# Initialize database
db = DBManager()
db.create_tables()

# Add a word entry
db.insert_or_update_word(
    term="algorithm",
    definition="A step-by-step procedure for solving a problem",
    part_of_speech="noun"
)

# Create relationships
db.insert_relationship("algorithm", "procedure", "synonym")
```

### Command Line Interface

The package installs a `word_forge` executable:

```bash
# Show version
word_forge --version

# Start the processing pipeline with seed words
word_forge start apple banana --minutes 5 --workers 4

# Build the semantic graph
word_forge graph build --timeout 180

# Generate visualization (requires visualization extra)
word_forge graph visualize --3d --open-browser

# Index vectors
word_forge vector index --embedder MiniLM-L6-v2

# Annotate emotions
word_forge emotion annotate --strategy hybrid

# Run full demo pipeline
word_forge demo full --3d --open-browser

# Setup NLTK data
word_forge setup-nltk

# Quiet mode (suppress non-error output)
word_forge --quiet start apple

# Verbose mode (enable debug output)
word_forge --verbose start apple
```

### Demo Scripts

```bash
# Explore configuration
python -m word_forge.demos.config_demo --validate

# Vector worker demo
python -m word_forge.demos.vector_worker_demo

# Generate lexical data for a word
python lexical_proto.py recursion

# Database demo
python -m word_forge.demos.database_demo
```

## Project Structure

```
word_forge/
├── src/word_forge/        # Main package
│   ├── config.py          # Central configuration
│   ├── forge.py           # CLI entry point
│   ├── configs/           # Configuration components
│   ├── database/          # SQLite persistence layer
│   ├── emotion/           # Emotion analysis system
│   ├── graph/             # Semantic graph operations
│   ├── parser/            # Text parsing and lexical extraction
│   ├── queue/             # Worker queue management
│   ├── vectorizer/        # Vector embeddings and search
│   └── demos/             # Example scripts
├── tests/                 # Test suite (pytest)
├── docs/                  # Documentation
├── data/                  # Data directory (created at runtime)
└── pyproject.toml         # Project configuration
```

## Development

### Code Style

Word Forge uses **Black** for formatting and **Ruff** for linting:

```bash
# Format code
black .

# Lint code
ruff check .

# Auto-fix lint issues
ruff check . --fix
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_config.py

# Run with coverage
pytest --cov=word_forge --cov-report=html
```

### Pre-commit Hooks

Install pre-commit hooks to automatically format and lint on commit:

```bash
pip install pre-commit
pre-commit install
```

## Configuration

Word Forge uses a centralized configuration system with environment variable overrides:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `WORDFORGE_DB_PATH` | Database file path | `data/word_forge.sqlite` |
| `WORDFORGE_LOG_LEVEL` | Logging level | `INFO` |
| `WORDFORGE_VECTOR_MODEL` | Embedding model name | `all-MiniLM-L6-v2` |

Configuration can also be modified programmatically:

```python
from word_forge.config import config

# Access configuration
print(config.database.db_path)
print(config.vectorizer.model_name)

# Export configuration
config.export_to_file("config.json")
```

## NLTK Data

Word Forge uses NLTK for WordNet and other linguistic resources. Data is downloaded automatically on first use, or can be pre-downloaded:

```bash
word_forge setup-nltk
```

Required corpora: WordNet, Punkt, stopwords, VADER lexicon.

## Documentation

- [`docs/overview.md`](docs/overview.md) - Developer guide
- [`docs/glossary.md`](docs/glossary.md) - Term definitions
- [`docs/templates/`](docs/templates/) - Docstring templates

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes and add tests
4. Ensure tests pass (`pytest`)
5. Ensure code is formatted (`black . && ruff check .`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

All pull requests must pass CI checks before merging.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NetworkX](https://networkx.org/) - Graph operations
- [NLTK](https://www.nltk.org/) - Natural language processing
- [Sentence Transformers](https://www.sbert.net/) - Semantic embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database
