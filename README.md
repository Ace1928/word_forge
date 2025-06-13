# Word Forge

Word Forge is an experimental lexical data processing framework. It aggregates natural language resources, analyzes emotional context, and builds a semantic graph for advanced text exploration.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Modules are located under `src/word_forge`. Example usage:

```python
from word_forge.config import config

print(config.database_url)
```

## Contributing

- Format code with `black`.
- Lint with `ruff`.
- Run tests with `pytest`.

Pull requests should pass all checks in the CI workflow.
