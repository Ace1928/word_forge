# Configuration and storage

## Runtime directories

Word Forge never needs to write into its installed Python package. Defaults
follow each operating system's application-data convention.

Environment variables, highest precedence first:

| Variable | Meaning |
| --- | --- |
| `WORD_FORGE_HOME` | Move all Word Forge runtime state |
| `WORD_FORGE_DATA_DIR` | Override databases, indexes, exports, and graphs |
| `WORD_FORGE_LOG_DIR` | Override logs |
| `WORD_FORGE_MODELS_DIR` | Override local GGUF and model tools |
| `WORD_FORGE_DB_PATH` | Select the exact default SQLite database file |

On Linux, `XDG_DATA_HOME` is honored when `WORD_FORGE_HOME` is unset.

## Optional service URLs

`WORD_FORGE_OLLAMA_HTTP_URL` configures an Ollama-compatible host. A pathless
value such as `http://model-host:11434` receives the correct generation or
embedding API path for each call. `OLLAMA_HOST` is accepted as a compatibility
fallback.

## Call timing

- `WORD_FORGE_TRACE_CALLS=1` logs decorated call durations at debug level.
- `WORD_FORGE_SLOW_CALL_MS=250` warns when a decorated call takes at least 250
  milliseconds.

Arguments and results are never included in these timing logs.

## Configuration files

The advanced CLI accepts `--config FILE` with UTF-8 JSON or TOML. Unknown
components and fields are rejected, and a file is validated before any live
setting changes. Prefer environment variables for deployment paths and explicit
`Lexicon(path)` values when an application owns its data location.
