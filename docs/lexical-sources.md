# Lexical source governance

Word Forge code is MIT licensed. Imported lexical records retain the terms of
their original datasets. The machine-readable registry in
`word_forge.sources` keeps those two concerns separate and gives importers one
policy boundary to enforce.

Run `word_forge sources list` for a compact catalog or
`word_forge sources list --json` for automation. The
`--unattended-eligible` filter includes only the core and permissive tiers; it
does not mean a planned importer is already implemented.

This catalog is operational guidance, not legal advice. Notices and license
metadata shipped with the exact downloaded snapshot are authoritative.

## Source tiers

| Tier | Automated behavior | Data placement |
| --- | --- | --- |
| `core` | May be used by the lightweight bootstrap | Local runtime data with required notices |
| `permissive` | Eligible for a future unattended opt-in download | Versioned local snapshot with provenance |
| `share-alike-opt-in` | Requires an explicit license-aware choice | Separate data directory and attribution manifest |
| `per-dataset-review` | Requires inspection of every selected component | Separate snapshot after its license is recorded |
| `external-runtime` | Never vendored by the MIT package | Invoke a separately installed program or library |

## Primary sources

- [Princeton WordNet](https://wordnet.princeton.edu/license-and-commercial-use)
  permits commercial use under its notice and disclaimer requirements. Word
  Forge currently accesses it through NLTK.
- [CMUdict](https://github.com/cmusphinx/cmudict/blob/master/LICENSE) permits
  unrestricted research and commercial use and requests acknowledgement of
  Carnegie Mellon University. It is the current source-backed English
  pronunciation provider.
- [PanLex](https://panlex.org/license) publishes its CSV and JSON snapshots
  under CC0 and requests a citation. It is a strong permissive candidate for
  broad translation coverage.
- [Wikidata Lexemes](https://dumps.wikimedia.org/legal.html) are structured
  Lexeme-namespace data under CC0. Text in other Wikimedia namespaces has
  different terms and must not be silently mixed into the CC0 layer.
- [Unicode data and software](https://www.unicode.org/copyright.html) use the
  Unicode License v3 unless a release says otherwise. Stable
  [CLDR releases](https://cldr.unicode.org/index/downloads) should be pinned.
- [DBnary](https://kaiko.getalp.org/about-dbnary/) extracts structured lexical
  data from Wiktionary and distributes data under attribution/share-alike
  terms. It belongs in a separate, explicit data layer.
- [Kaikki](https://kaikki.org/dictionary/rawdata.html) provides large,
  frequently refreshed Wiktionary extracts. Importers must retain originating
  Wiktionary provenance and the exact extract's notices; Wiktextract's software
  license does not replace the content license.
- [UniMorph](https://unimorph.github.io/) publishes morphology by language.
  Each language repository declares its own source and license, so Word Forge
  must evaluate them individually.
- [Open Multilingual Wordnet](https://omwn.org/) aggregates independently
  licensed wordnets. Importers must preserve each component's license and
  citation rather than applying one inferred license to the aggregate. Word
  Forge selects the corpus package expected by the installed NLTK line
  (`omw-1.4` for NLTK 3.9 and `omw-2.0` for NLTK 3.10+) and requires both
  `--multilingual` and `--accept-source-licenses`; it is excluded from
  unattended bootstrap.
- [Epitran](https://github.com/dmort27/epitran) provides MIT-licensed,
  language-and-script-specific orthography-to-IPA mappings.
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) offers compact phoneme
  rules for many languages under GPLv3-or-later. Word Forge treats it as an
  external runtime and does not vendor it into the MIT package.

## Required provenance envelope

Every downloaded snapshot and every imported assertion should retain enough
metadata to reproduce and audit it:

1. stable source identifier and upstream record identifier;
2. exact source URL, release or dump timestamp, and retrieval time;
3. SHA-256 digest and byte size of the downloaded artifact;
4. license identifier, license URL, notices, and required attribution;
5. importer name and version plus transformations performed;
6. language and script, confidence, and whether a value was generated;
7. the local snapshot ID that produced each lexical assertion.

Downloads should be streamed to a temporary file, bounded by an explicit size
limit, verified against a recorded digest when upstream publishes one, and
atomically promoted only after validation. Import should be transactional and
idempotent so an interrupted bootstrap can resume without duplicated records.
