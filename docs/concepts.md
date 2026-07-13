# Core concepts

## Lexicon

A lexicon is more than a word list. Each entry can contain a meaning, grammar,
examples, language, origin, pronunciation information, emotional dimensions,
and connections to other entries.

## Word and lexeme

A word is the familiar written form used in the base-language layer. A lexeme
is a language-specific vocabulary item. Separating them prevents two words that
look alike in different languages from being treated as the same thing.

For example, English `gift` and German `Gift` have different meanings. Their
language tags keep them distinct.

## Relationship

A relationship is a labeled arrow between entries:

```text
sparrow --is_a--> bird
hot --antonym--> cold
house [en] --translation--> casa [es]
```

Labels are data, so a project may use standard linguistic labels or define its
own domain-specific ones.

## Semantic graph

When all relationships are viewed together, the lexicon becomes a graph. Graph
algorithms can find neighborhoods, clusters, central concepts, paths, and gaps
that a flat dictionary cannot reveal.

## Embedding

An embedding is a numeric representation produced by a neural model. Terms with
similar usage tend to have nearby embeddings. Embeddings are optional indexes;
the readable SQLite records remain the source of truth.

## Living lexicon

"Living" means entries can be incrementally enriched. New sources add evidence,
workers connect records, quality checks identify weak entries, and indexes are
rebuilt without replacing the core data model.

## Sense and revision

A spelling can have several senses. For example, `bank` may describe a financial
institution or the side of a river. Word Forge stores these meanings separately
so each can carry its own source, confidence, review state, and revision.

A revision is a counter that changes when a record changes. Applications can
send the revision they read; Word Forge then refuses an update if someone else
has already saved a newer version.

## Local-first

The default database and generated files remain on the user's machine. Word
Forge does not require an account or hosted service. Optional remote model
providers are used only when configured and invoked.
