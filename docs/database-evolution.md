# Database evolution and recovery

Word Forge upgrades an older SQLite lexicon automatically when it is opened.
Migrations are ordered, transactional, and recorded with SQLite's schema
version. A failed migration rolls back instead of leaving a partly upgraded
database, and a database created by a newer Word Forge release is rejected
instead of being silently downgraded.

## Durable records

Words, lexemes, translations, relationships, and senses have stable UUIDs and
monotonic revisions. Updates retain identity. The change log records each
creation or update, while an invalidation queue tells derived graph, search, and
vector views what must be refreshed.

Senses are independent records. Revising one meaning does not rewrite every
meaning of the same spelling. Removal creates a tombstone so audit history can
be retained.

## Safe upgrade procedure

1. Keep normal readers open if needed, but stop bulk writers and enrichment workers.
2. Run `word-forge integrity` against the current database.
3. Create a consistent online snapshot with `word-forge backup`; do not copy a
   live SQLite file by itself because committed data may still be in its WAL.
4. Install the new Word Forge release in a fresh virtual environment.
5. Open a restored copy first and verify representative entries and statistics.
6. Keep the snapshot until downstream indexes have been rebuilt and checked.

The CLI can perform the consistency-sensitive steps while the source database
is live:

```bash
word-forge integrity --db-path lexicon.sqlite
word-forge backup backups/lexicon.sqlite --db-path lexicon.sqlite
word-forge restore backups/lexicon.sqlite --db-path restored.sqlite
```

An existing backup or restore destination is never replaced without `--force`.

SQLite is the source of truth. Vector indexes, graph layouts, reports, and web
assets can be regenerated and do not need to be restored with the database.

## Concurrent editors

`Lexicon.add(..., expected_revision=N)` performs an optimistic revision check.
Pass `0` to require a new entry or pass the revision returned by `get()` to
prevent a stale editor from overwriting newer work. Conflicts raise
`RevisionConflictError`; the local HTTP API returns status `409` with both
revision values.
