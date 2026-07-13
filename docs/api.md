# Python API

## Recommended entry point

```python
from word_forge import Lexicon
```

`Lexicon(path=None)` opens the default database or an explicit SQLite file. It
is a context manager, so pooled connections are always released:

```python
with Lexicon("terms.sqlite") as lexicon:
    lexicon.add("graph", definition="Nodes connected by edges.")
```

## Stable methods

- `add(...) -> LexicalEntry`: validate and upsert a language-specific entry;
  pass `expected_revision` to reject stale writes.
- `get(term, language="en") -> LexicalEntry`: retrieve one exact entry.
- `find(query, language=None, limit=20) -> list[LexicalEntry]`: bounded text
  search with literal handling for `%` and `_`.
- `relate(term, target, kind="related") -> bool`: add a base-language edge.
- `translate(...) -> None`: add one-way or bidirectional translation records.
- `add_sense(...) -> Sense`: create or revise an independently versioned
  meaning, domain, register, source, confidence, and review state.
- `remove_sense(...) -> bool`: tombstone one sense while preserving history.
- `statistics() -> LexiconStatistics`: count records using constant-memory SQL.
- `jobs -> JobQueue`: access crash-recoverable background work using the same
  transaction store.
- `close()`: release resources; safe to call more than once.

All returned data models are frozen, slotted dataclasses. `LexicalEntry`
includes its stable UUID, normalized term, detected script, senses, revision,
confidence, review state, and timestamps. `as_dict()` produces a JSON-compatible
copy.

## Exceptions

- `ValueError`: malformed or unsafe user input.
- `TermNotFoundError`: an exact requested entry does not exist.
- `DatabaseError`: storage or transaction failure.
- `RevisionConflictError`: an optimistic update used a stale revision.
- `RuntimeError`: an operation was attempted after `close()`.

The deeper packages under `word_forge.database`, `word_forge.graph`,
`word_forge.multilingual`, `word_forge.emotion`, and `word_forge.vectorizer`
remain available for specialized applications. Their APIs are more detailed and
may evolve faster than the high-level `Lexicon` contract.
