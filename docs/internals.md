# Internal module map

Most applications need only `word_forge.Lexicon`. The modules below explain
where maintainers should make deeper changes.

| Area | Responsibility |
| --- | --- |
| `lexicon.py` | Stable validation, storage, search, relationship, translation, sense, and network facade |
| `models.py` | Immutable public return types and JSON conversion |
| `database/` | SQLite schema, transactions, migrations, durable identities, revision checks, audit history, and jobs |
| `multilingual/` | Language normalization, batch ingestion, source catalogs, and repair |
| `linguistics/` | Morphology, phonetics, grapheme-to-phoneme conversion, and prosody |
| `phrases/` | Multi-word expressions and component ordering |
| `graph/` | NetworkX construction, algorithms, import/export, and optional visualization |
| `vectorizer/` | Optional neural embeddings and memory, SQLite, Chroma, FAISS, or HNSW indexes |
| `emotion/` | Dimensional and categorical affect processing |
| `parser/` | Algorithmic NLP, WordNet enrichment, structured output validation, and model adapters |
| `queue/` | Bounded worker queues, retries, and lifecycle management |
| `jobs.py` | Durable idempotent jobs, atomic claims, expiring leases, retry limits, and cancellation |
| `conversation/` | Conversation ingestion and enrichment workflows |
| `web/` | Loopback FastAPI API, session protection, live invalidation events, and packaged browser assets |
| `bridge/` | Optional read-only audits of documented SQLite/JSON interchange data |
| `paths.py` | Cross-platform user data, model, log, and report locations |
| `observability.py` | Opt-in timing without argument or result logging |

## Dependency direction

The stable facade depends inward on storage and immutable models. Optional
providers are imported only inside the operation that uses them. No module
imports code from the former monorepo or from a sibling checkout.

## Adding a subsystem

Keep its public boundary typed, validate inputs before side effects, make file
locations configurable, use bounded work, and provide an offline fallback or a
clear optional-install message. Add unit tests and update the architecture,
configuration, security, and performance guides when the new behavior affects
them.
