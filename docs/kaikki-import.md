# Kaikki/Wiktionary import

`word_forge.sources.kaikki` streams Wiktextract JSON Lines into the normalized
lexical schema. Plain JSONL, gzip, and bzip2 artifacts are supported without
extracting the whole corpus or holding it in memory.

Kaikki data is derived from Wiktionary. Import requires an explicit source
license acknowledgement and stores that fact with the source snapshot. Users
remain responsible for attribution, share-alike, and any notices shipped with
the exact artifact. See [lexical source governance](lexical-sources.md).

## Safe import sequence

The importer deliberately separates artifact inspection from mutation:

```python
from pathlib import Path

from word_forge.database.database_manager import DBManager
from word_forge.database.lexical_repository import LexicalRepository
from word_forge.sources.kaikki import KaikkiImporter, inspect_artifact

database = DBManager("data/word_forge.sqlite")
artifact = inspect_artifact(Path("data/kaikki-en.jsonl.bz2"))
importer = KaikkiImporter(
    LexicalRepository(database),
    source_version="2026-07-06",
    source_url="https://kaikki.org/dictionary/rawdata.html",
    accept_source_license=True,
    batch_size=500,
    languages=("en",),
)
report = importer.import_artifact(
    artifact,
    checkpoint_path=Path("data/checkpoints/kaikki-en.json"),
)
```

Inspection computes SHA-256 over the exact compressed or plain source bytes in
bounded memory. The importer verifies size and modification time again before
and after processing, registers the digest in `source_snapshots`, and writes
entries in bounded transactions. Do not modify or replace an artifact while an
import is running.

## Resuming

After each committed batch, the importer atomically writes a versioned
checkpoint containing the artifact digest, snapshot ID, next source line, and
total imported entries. Reusing the checkpoint skips already committed lines.
For compressed inputs this is functionally resumable but seeking still requires
decompression from the start; plain JSONL is preferable when frequent resumes
are expected.

Malformed JSON or lexical records stop the import by default. The raised
`KaikkiImportError` reports both the failing line and the last committed line.
Earlier batches remain valid and the checkpoint points to the first uncommitted
line.

## Preserved fields

The current importer preserves:

- lemma, language, script, part of speech, lexical category, and etymology;
- alternate and inflected forms with tags/features;
- distinct senses with normalized and raw glosses;
- examples, references, and identified translations;
- IPA/English-pronunciation records and secure audio URLs;
- entry- and sense-level synonyms, antonyms, hypernyms, hyponyms, holonyms,
  meronyms, troponyms, coordinate terms, derivations, related terms, and
  translations;
- selected source categories, topics, external concept links, and the exact
  source line in canonical JSON metadata.

Unrecognized source fields remain available in the immutable original artifact
identified by the snapshot digest. Importer revisions can replay that artifact
without changing or duplicating records from an earlier snapshot.
