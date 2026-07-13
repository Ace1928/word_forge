# Command-line reference

Run `word-forge --help` for the installed version's complete list. Every command
returns `0` on success. Input or database errors return `2`; an exact entry that
does not exist returns `3`; an optimistic revision conflict returns `4`.

## Everyday commands

| Command | Purpose |
| --- | --- |
| `word-forge init` | Create the database schema |
| `word-forge add TERM` | Add or update one entry |
| `word-forge show TERM` | Show an exact entry (`lookup` is an alias) |
| `word-forge find QUERY` | Search terms and definitions (`search` is an alias) |
| `word-forge relate A B` | Add a typed relationship |
| `word-forge translate A B` | Add a language-aware translation |
| `word-forge sense TERM DEFINITION` | Add or revise an independently versioned meaning |
| `word-forge delete-sense UUID` | Tombstone one meaning while retaining history |
| `word-forge stats` | Show exact record counts |
| `word-forge integrity` | Check SQLite integrity, foreign keys, and lifecycle queues |
| `word-forge backup FILE` | Create a consistent online snapshot |
| `word-forge restore FILE` | Restore a healthy snapshot; overwrite requires `--force` |
| `word-forge doctor` | Check required and optional features |
| `word-forge gui` | Open the authenticated local visual workspace (`serve` is an alias) |

All accept `--db-path FILE` when they operate on a lexicon. Commands that print
records accept `--json` for scripts.

Use `--if-revision 0` to require creation or `--if-revision N` to update only
the revision you previously read. Backup and restore refuse to replace existing
files unless `--force` is explicit.

## Enrichment and analysis

| Command | Purpose |
| --- | --- |
| `word-forge setup-nltk` | Download NLTK lexical datasets |
| `word-forge start WORD...` | Run bounded worker-based enrichment |
| `word-forge graph build` | Rebuild semantic graph state |
| `word-forge graph visualize` | Generate an interactive HTML graph |
| `word-forge vector index` | Build the optional neural index |
| `word-forge vector search QUERY` | Query the neural index |
| `word-forge emotion annotate` | Annotate stored entries with emotion data |

Use `--with-vector` or `--with-llm` on `start` to opt into model-powered work.
These flags may cause the configured provider to download a model.

## Automation example

```bash
word-forge stats --json | python -c \
  "import json,sys; print(json.load(sys.stdin)['lexemes'])"
```

JSON is emitted as UTF-8 with stable field names and sorted object keys.

## Local workspace

Install `word-forge[gui]`, then run `word-forge gui`. The server accepts only
loopback hosts. Use `--db-path`, `--port`, or `--no-open` when an
application manages the database, port, or browser lifecycle.
