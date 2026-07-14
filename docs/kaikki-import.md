# Kaikki/Wiktionary import

`word_forge.sources.kaikki` streams Wiktextract JSON Lines into the normalized
lexical schema. Plain JSONL, gzip, and bzip2 artifacts are supported without
extracting the whole corpus or holding it in memory.

Kaikki data is derived from Wiktionary. Import requires an explicit source
license acknowledgement and stores that fact with the source snapshot. Users
remain responsible for attribution, share-alike, and any notices shipped with
the exact artifact. See [lexical source governance](lexical-sources.md).

## Safe import sequence

The CLI hashes the artifact before opening the database and uses a resumable
sidecar checkpoint by default:

```bash
word_forge data import-kaikki data/raw-wiktextract-data.jsonl.gz \
  --source-version 2026-07-06 \
  --source-url https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz \
  --expected-sha256 EXACT_64_HEX_DIGEST \
  --accept-source-license \
  --language en \
  --db-path data/word_forge.sqlite
```

Repeat `--language BCP47` to select more languages. Omit every language option
to import the whole artifact. Use `--max-entries` for a staged run, `--json`
for an automation report, `--checkpoint PATH` to relocate the checkpoint, or
`--no-checkpoint` only when resume support is deliberately unnecessary.

The Python API also separates artifact inspection from mutation:

```python
from pathlib import Path

from word_forge.database.database_manager import DBManager
from word_forge.database.lexical_repository import LexicalRepository
from word_forge.sources.kaikki import KaikkiImporter, inspect_artifact

database = DBManager("data/word_forge.sqlite")
artifact = inspect_artifact(
    Path("data/raw-wiktextract-data.jsonl.gz"),
    expected_sha256="EXACT_64_HEX_DIGEST",
)
importer = KaikkiImporter(
    LexicalRepository(database),
    source_version="2026-07-06",
    source_url="https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz",
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
checkpoint containing the artifact digest, import-configuration digest,
snapshot ID, next source line, and total imported entries. The configuration
digest binds the checkpoint to its database path, source metadata, importer
format, and language filters. Reusing a checkpoint with different settings is
rejected before source-snapshot mutation instead of silently skipping records.
Reusing a matching checkpoint skips already committed lines.
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
