# Maintainer scripts

These scripts measure release behavior; they do not modify a user's lexicon.
Run them from the repository root after installing the development environment.

## Offline core benchmark

```bash
python scripts/benchmark_core.py
python scripts/benchmark_core.py --entries 2000 --searches 500 --json
```

This benchmark creates and deletes an isolated temporary database. It exercises
entry creation, relationships, exact lookup, text search, and bounded network
generation without accessing the internet or downloading a model.

## Optional model benchmark

```bash
python scripts/benchmark_llm_backends.py --model ollama:llama3.2
```

The model benchmark invokes the named provider and may send prompts to it or
download model data according to that provider's behavior. Do not run it with a
remote provider unless that transmission is acceptable. Use `--json-out FILE`
to retain a structured comparison report.

Normal users do not need these scripts. Use `word-forge doctor` for a local
capability check and `word-forge start` for bounded enrichment work.
