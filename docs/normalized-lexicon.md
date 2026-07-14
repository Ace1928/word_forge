# Normalized lexical storage

Schema version 3 preserves source structure instead of flattening every meaning
of a spelling into one definition string. Existing callers can continue using
the `words` table and `DBManager`; importers use `LexicalRepository` and the
validated records in `word_forge.lexicon`.

```text
source snapshot
└── lexical entry ──> compatibility word
    ├── forms
    ├── senses
    │   ├── glosses
    │   └── examples
    ├── pronunciations ──> optional form
    └── relations ──> textual/source target
```

The compatibility word is a language-aware search and graph facade. Several
source entries may point to it, so homographs, parts of speech, etymologies,
and independently sourced analyses remain distinct in the normalized layer.

## Provenance

Every entry belongs to a `source_snapshots` row. A snapshot records the source
and release identifiers, upstream URL, retrieval time, artifact SHA-256 and
size when applicable, license and attribution, importer version, and canonical
JSON metadata. The tuple of source, version, and digest is idempotent.

The exact artifact's bundled license remains authoritative. The source policy
catalog can be inspected with `word_forge sources list --json`.

## Atomic writes

`LexicalRepository.upsert_entries()` writes one batch in a SQLite transaction.
It validates duplicate source identities before opening the transaction,
upserts the compatibility word and entry header, then replaces that entry's
children as one unit. An exception rolls back the whole batch. Replaying the
same snapshot is therefore deterministic and does not duplicate nested rows.

For large imports, callers should choose bounded batches. A completed batch is
durable; a failed batch is safe to retry. Import-level checkpoints can record
the last source offset after each successful batch.

## JSON fields

Tags, feature bundles, and source-specific metadata use canonical UTF-8 JSON:
keys are sorted, insignificant whitespace is removed, and non-finite numbers
are rejected. Common query identities—lemma/form normalization, language,
script, position, source IDs, and relation targets—remain ordinary indexed
columns rather than opaque JSON.

## Migration

Opening a schema-v2 database creates the normalized tables and indexes in one
validated transaction, preserves existing words and relationships, and moves
the SQLite application version to 3. Legacy schema-v1 rebuilding remains
lossless and also creates the v3 tables. Databases with a newer version are
rejected without mutation.
