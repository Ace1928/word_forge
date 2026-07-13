# Word Forge

[![CI](https://github.com/Ace1928/word_forge/actions/workflows/ci.yml/badge.svg)](https://github.com/Ace1928/word_forge/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Word Forge is a local-first multilingual living lexicon. It stores words and
phrases, definitions, examples, translations, semantic relationships, emotional
dimensions, graph structure, and optional neural embeddings in one inspectable
system.

In plain language: Word Forge lets you build a dictionary that understands how
its entries connect. It works offline for core use, keeps your data on your
computer, and adds model-powered enrichment only when you explicitly enable it.

## Why Word Forge

- Multilingual by design: store language-tagged lexemes and bidirectional
  translations without flattening them into one language.
- Highly connected: model synonyms, antonyms, hypernyms, hyponyms, morphology,
  phonetics, phrases, emotion, and custom relationship types.
- Meaning-aware: keep several independently sourced and reviewed senses for one
  spelling without flattening them into a single definition.
- Local and portable: SQLite is the source of truth. A lexicon is a normal file
  that can be backed up, copied, inspected, and versioned.
- Durable and conflict-aware: stable record identities, transactional schema
  upgrades, audit history, and optimistic revisions protect long-lived data.
- Useful without a GPU: manual editing, algorithmic NLP, graph operations, and
  search have a lightweight install.
- Neural when wanted: sentence-transformer search, local GGUF models,
  Hugging Face models, and Ollama-compatible services are optional.
- Automation friendly: every approachable CLI command has stable exit codes and
  JSON output.
- A real Python library: the `Lexicon` API is small, typed, documented, and
  independent of the original Eidosian Forge monorepo.

## Five-minute start

Word Forge supports Python 3.10 through 3.13.

```bash
git clone https://github.com/Ace1928/word_forge.git
cd word_forge
python -m venv .venv
```

Activate the environment:

```bash
# Linux and macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install and create a lexicon:

```bash
python -m pip install --upgrade pip
python -m pip install .
word-forge doctor
word-forge init
```

Add and connect entries:

```bash
word-forge add recursion \
  --part-of-speech noun \
  --definition "A process that refers to itself." \
  --example "Recursion can traverse a tree."

word-forge add casa --language es \
  --definition "Edificio para habitar." \
  --base-term house

word-forge translate house Haus --from en --to de
word-forge relate algorithm procedure --kind synonym
```

Find and inspect data:

```bash
word-forge show recursion
word-forge find process
word-forge find edificio --language es
word-forge stats
word-forge stats --json
```

No model is downloaded by these commands.

Create a consistent snapshot whenever the lexicon becomes valuable:

```bash
word-forge integrity
word-forge backup lexicon-backup.sqlite
```

Open the full local workspace:

```bash
python -m pip install ".[gui]"
word-forge gui
```

The server binds only to loopback and generates a new browser-session secret on
every launch. The responsive workspace includes structured entry views and a
coordinated live network explorer; the graph is never the only way to access
information.

## Python API

```python
from word_forge import Lexicon

with Lexicon("research.sqlite") as lexicon:
    lexicon.add(
        "emergence",
        definition="Complex behavior arising from simple interactions.",
        part_of_speech="noun",
        examples=["Flocking is an example of emergence."],
    )
    lexicon.relate("emergence", "complexity", kind="associated_with")
    lexicon.add_sense(
        "emergence",
        "A property present at one scale but not its individual components.",
        domain="systems science",
        source="research notes",
    )

    entry = lexicon.get("emergence")
    print(entry.definition)
    print(entry.as_dict())
```

`Lexicon` accepts an explicit SQLite path. If omitted, Word Forge uses the
operating system's normal per-user application-data directory.

## Core and optional installs

The default install is dependency-free and contains the complete SQLite-backed
lexicon, multilingual records, senses, provenance, revisions, audit history,
search, bounded network snapshots, backup, restore, and CLI. It imports in a
fraction of a second and works offline.

Install only the feature groups you need:

```bash
python -m pip install ".[nlp]"            # NLTK, NumPy, validation
python -m pip install ".[graph]"          # NetworkX and community analysis
python -m pip install ".[gui]"            # local visual workspace
python -m pip install ".[emotion]"
python -m pip install ".[vector]"
python -m pip install ".[visualization]"
python -m pip install ".[llm]"
python -m pip install ".[linguistics,metrics]"
python -m pip install ".[full]"
```

For the local visual workspace:

```bash
python -m pip install ".[gui]"
word-forge gui
```

The workspace binds to the loopback interface, opens in the default browser,
and renders the live multilingual network without external JavaScript, fonts,
or telemetry. `word-forge serve` remains an alias for scripts written against
the early workspace preview.

Then prepare NLTK's open lexical datasets when you want automatic WordNet and
sentiment enrichment:

```bash
python -m pip install ".[nlp]"
word-forge setup-nltk
```

Neural processing is opt-in at runtime:

```bash
word-forge start language knowledge --minutes 2 --with-vector
word-forge start language knowledge --minutes 2 --with-llm
```

These modes may download model files. `word-forge doctor` reports which optional
features are currently installed.

## Where data goes

| Platform | Default home |
| --- | --- |
| Linux | `$XDG_DATA_HOME/word-forge`, or `~/.local/share/word-forge` |
| macOS | `~/Library/Application Support/WordForge` |
| Windows | `%LOCALAPPDATA%\WordForge` |

Set `WORD_FORGE_HOME` to make an installation portable or place it on another
drive. `WORD_FORGE_DATA_DIR`, `WORD_FORGE_LOG_DIR`, and
`WORD_FORGE_MODELS_DIR` provide narrower overrides. `WORD_FORGE_DB_PATH`
selects one exact default SQLite file.

## Architecture at a glance

```text
CLI / Lexicon API
       |
       v
Validated domain services
       |
       +--> SQLite lexicon and multilingual records
       +--> bounded dependency-free network snapshots
       +--> optional NetworkX analysis and exports
       +--> emotion and phrase processing
       +--> optional vector indexes and model providers
       |
       v
Typed results, JSON, graphs, and portable exports
```

Heavy libraries are imported lazily. Optional failures produce installation
hints rather than making core lexicon operations unusable.

## Documentation

- [Getting started](docs/getting-started.md)
- [Concepts for non-technical readers](docs/concepts.md)
- [Command-line reference](docs/cli.md)
- [Python API guide](docs/api.md)
- [Visual workspace guide](docs/gui.md)
- [Configuration and data locations](docs/configuration.md)
- [Architecture and data flow](docs/architecture.md)
- [Performance and scale](docs/performance.md)
- [Security and privacy](docs/security.md)
- [Internal module map](docs/internals.md)
- [Database evolution and recovery](docs/database-evolution.md)
- [Contributing guide](CONTRIBUTING.md)

## Quality and support

The repository checks supported Python versions on Linux, macOS, and Windows,
runs unit and integration tests, lints and type-checks stable public modules,
builds both wheel and source distributions, and smoke-tests the built wheel in a
clean environment.

For bugs and feature requests, use [GitHub Issues](https://github.com/Ace1928/word_forge/issues).
Security reports follow [SECURITY.md](SECURITY.md). Development expectations are
in [CONTRIBUTING.md](CONTRIBUTING.md).

Word Forge is licensed under the [MIT License](LICENSE).
