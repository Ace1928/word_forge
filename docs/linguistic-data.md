# Linguistic data model

Word Forge keeps written form, pronunciation, language, and meaning as
separate concepts. This avoids treating a Unicode code point as a user-visible
character or treating spelling as pronunciation.

## Language identity

Language values use structurally valid BCP 47 tags with conventional casing:
language subtags are lowercase, ISO 15924 script subtags are title case, and
region subtags are uppercase. Word Forge validates tag structure locally; it
does not claim that every structurally valid subtag is currently registered by
IANA. See [RFC 5646](https://www.rfc-editor.org/rfc/rfc5646.html).

Lexical identity uses a Unicode NFKC, case-folded lookup key while retaining the
original display spelling. Language remains part of identity because the same
spelling can represent unrelated words in different languages.

## Graphemes

`segment_graphemes()` implements Unicode extended grapheme-cluster boundaries
through the `regex` package's `\X` support. Every record retains:

- original and NFC-normalized cluster text;
- scalar values such as `U+0065`;
- Unicode character names, general categories, and combining classes;
- an inferred ISO 15924 script code.

This follows [Unicode Standard Annex #29](https://unicode.org/reports/tr29/).
Emoji sequences, regional-indicator flags, and base-plus-combining-mark text
therefore remain single grapheme clusters where the standard requires it.

## Pronunciations and phonemes

Pronunciations always identify notation, language, dialect, source, confidence,
and whether they were derived. The initial offline source is NLTK's CMU
Pronouncing Dictionary, an American-English resource. It provides stress-marked
ARPABET records; Word Forge also exposes a deterministic approximate IPA
conversion marked as generated and with reduced confidence. See the
[NLTK CMUdict reader](https://www.nltk.org/_modules/nltk/corpus/reader/cmudict.html).

Non-English lookups return no pronunciation until a language-appropriate source
or grapheme-to-phoneme provider is configured. They are never silently routed
through the English dictionary. This boundary lets future DBnary, Wiktionary,
language-specific dictionary, and optional neural G2P imports coexist without
losing provenance.
