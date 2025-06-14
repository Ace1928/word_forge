# Word Forge Developer Guide

This guide provides a high-level overview of Word Forge's components.

- **database** — manages the SQLite storage of lexical data.
- **emotion** — computes emotional metrics for words and passages.
- **graph** — builds a semantic network from processed terms.
- **parser** — extracts and refines lexical entries from raw text.
- **vectorizer** — generates numeric embeddings for terms and documents.
- **queue** — orchestrates worker tasks for asynchronous processing.

Refer to `upgrade_plan.md` for planned improvements and the `docs/migration` directory for historical notes.

See `glossary.md` for definitions of key terms. Docstring templates for new code reside in the `docs/templates` directory.

