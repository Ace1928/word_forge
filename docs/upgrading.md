# Upgrading from the experimental repository

The standalone 1.x line keeps the useful lexical, graph, parser, emotion, and
vector systems while removing generated data and private monorepo coupling.

Important changes:

- Runtime data now uses the operating system's application-data directory. Set
  `WORD_FORGE_HOME` to keep a portable directory.
- Core installation is lightweight. Install feature extras such as `vector`,
  `llm`, `visualization`, `emotion`, or `gui` only when needed.
- Neural vector and language-model enrichment is opt-in through
  `--with-vector` and `--with-llm`.
- Configuration files are strict JSON or TOML. Unknown fields and malformed
  values fail before runtime state changes.
- `word-forge` is the canonical command; `word_forge` remains an executable
  alias. `lookup`, `search`, and `serve` remain command aliases.
- Generated databases, indexes, graphs, model files, logs, profiles, editor
  settings, and local cookies are not repository content.

Existing SQLite lexicons are migrated automatically. Follow the backup process
in [Database evolution and recovery](database-evolution.md) before opening the
only copy of valuable data.
