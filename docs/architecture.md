# Architecture

## Design goals

Word Forge separates durable lexical facts from replaceable derived indexes.
SQLite records can survive graph-library, vector-model, and visualization
upgrades.

```text
User or application
        |
        v
CLI and high-level Lexicon API
        |
        v
Validation and normalization
        |
        +-------------------+
        |                   |
        v                   v
SQLite source of truth   Optional providers
        |                (NLTK, models, Ollama)
        v                   |
Graph / vector / emotion <--+
derived views
```

## Components

- `lexicon.py`: stable public facade and Unicode/language validation.
- `database/`: SQLite schema, transactions, connection lifecycle, migrations,
  words, lexemes, translations, phrases, morphology, and phonetics.
- `multilingual/`: ingestion, normalization, source catalogs, aligned vectors,
  repair, and language-aware batch operations.
- `graph/`: NetworkX construction, queries, analysis, layouts, import/export,
  and optional visualization.
- `vectorizer/`: optional memory, SQLite, Chroma, FAISS, and HNSW vector paths.
- `emotion/`: categorical and dimensional affect processing with optional
  analyzers.
- `queue/`: bounded worker lifecycle and background processing.
- `jobs.py`: crash-recoverable SQLite job leases for durable background work.
- `parser/`: WordNet, structured validation, language-model adapters, and
  recursive enrichment.
- `web/`: optional loopback-only FastAPI workspace, strict request schemas,
  bounded live events, and self-contained static network rendering.

## Dependency boundaries

Core imports must succeed with only project dependencies. Neural, visualization,
transcription, and service integrations are optional and lazily imported. The
standalone package does not import files from Eidosian Forge or any sibling
repository.

## Persistence

SQLite foreign keys and write-ahead logging protect integrity and concurrent
reads. The database is the authoritative record. Vector indexes, HTML graphs,
reports, caches, and model files are derived runtime artifacts and are excluded
from version control and package distributions.

Schema upgrades are transactional and versioned. Stable UUIDs identify lexical
records independently of SQLite row numbers, optimistic revisions prevent stale
updates, and change/invalidation records make derived-index rebuilds explicit.
See [Database evolution](database-evolution.md) for recovery guidance.

## Durable background work

`JobQueue` stores work in the same authoritative SQLite database. Enqueue calls
carry an idempotency key, so retrying an identical request returns the original
job instead of duplicating it. A worker atomically claims one pending job for a
bounded lease. Heartbeats extend active leases; an expired lease becomes
claimable again until its retry limit is reached. Only the lease owner may
complete or fail running work.

This design makes process crashes recoverable without an external queue. Job
payloads must be JSON objects and should contain stable record UUIDs rather than
temporary row numbers. See [Durable jobs](jobs.md) for examples.
