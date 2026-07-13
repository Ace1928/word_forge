"""
Ingest multilingual lexicon data from Wiktextract/Kaikki JSONL dumps.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from word_forge.multilingual.multilingual_manager import MultilingualManager

LOGGER = logging.getLogger("word_forge.multilingual.ingest")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".gz":
        handle_ctx = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle_ctx = path.open("r", encoding="utf-8")
    with handle_ctx as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _normalize_language_filter(values: Sequence[str] | str | None) -> set[str]:
    if values is None:
        return set()
    items = values.split(",") if isinstance(values, str) else values
    return {str(item or "").strip().lower() for item in items if str(item or "").strip()}


def _entry_language(entry: dict[str, Any]) -> str:
    return str(entry.get("lang_code") or entry.get("lang") or "").strip().lower()


def _extract_gloss(entry: dict[str, Any], *, limit: int = 3) -> str:
    senses = entry.get("senses") or []
    gloss_rows: list[str] = []
    seen: set[str] = set()
    for sense in senses if isinstance(senses, list) else []:
        if not isinstance(sense, dict):
            continue
        for key in ("glosses", "raw_glosses", "tags"):
            values = sense.get(key) or []
            if not isinstance(values, list):
                continue
            for value in values:
                text = str(value or "").strip()
                if not text:
                    continue
                normalized = text.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                gloss_rows.append(text)
                if len(gloss_rows) >= max(1, int(limit or 3)):
                    return "; ".join(gloss_rows)
    return "; ".join(gloss_rows)


def _extract_usage_examples(entry: dict[str, Any], *, limit: int = 4) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    senses = entry.get("senses") or []
    for sense in senses if isinstance(senses, list) else []:
        if not isinstance(sense, dict):
            continue
        examples = sense.get("examples") or []
        if not isinstance(examples, list):
            continue
        for example in examples:
            if isinstance(example, dict):
                candidate = (
                    example.get("text") or example.get("english") or example.get("ref") or ""
                )
            else:
                candidate = example
            text = str(candidate or "").strip()
            if not text:
                continue
            normalized = text.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            rows.append(text)
            if len(rows) >= max(1, int(limit or 4)):
                return rows
    return rows


def _extract_translations(entry: dict[str, Any], *, limit: int = 24) -> Iterable[dict[str, Any]]:
    translations = entry.get("translations") or []
    emitted = 0
    if isinstance(translations, list):
        for t in translations:
            if not isinstance(t, dict):
                continue
            target_lang = str(t.get("lang_code") or t.get("lang") or "").strip().lower()
            target_term = str(t.get("word") or t.get("term") or t.get("translation") or "").strip()
            if not target_lang or not target_term:
                continue
            yield {
                "lang": target_lang,
                "term": target_term,
                "relation": str(t.get("relation") or "translation").strip() or "translation",
            }
            emitted += 1
            if emitted >= max(1, int(limit or 24)):
                return


def _multilingual_definition(*, lang: str, gloss: str, part_of_speech: str) -> str:
    normalized_lang = str(lang or "").strip().lower()
    normalized_pos = str(part_of_speech or "").strip().lower()
    normalized_gloss = str(gloss or "").strip()
    if normalized_gloss:
        prefix = " ".join(part for part in [normalized_lang, normalized_pos] if part).strip()
        if prefix:
            return f"{prefix} meaning: {normalized_gloss}"
        return normalized_gloss
    return f"{normalized_lang or 'multilingual'} lexical entry"


def ingest_wiktextract_jsonl(
    path: str,
    manager: MultilingualManager | None = None,
    base_lang: str = "en",
    limit: int | None = None,
    include_languages: Sequence[str] | str | None = None,
    exclude_languages: Sequence[str] | str | None = None,
    max_examples_per_entry: int = 4,
    max_translations_per_entry: int = 24,
    materialize_words: bool = True,
) -> None:
    """
    Ingest a Wiktextract JSONL dump.
    """
    manager = manager or MultilingualManager()
    path_obj = Path(path)
    if not path_obj.exists():
        LOGGER.warning("Wiktextract file not found: %s", path)
        return

    include_langs = _normalize_language_filter(include_languages)
    exclude_langs = _normalize_language_filter(exclude_languages)
    count = 0
    for entry in _iter_jsonl(path_obj):
        lemma = str(entry.get("word") or "").strip()
        lang = _entry_language(entry)
        if not lemma or not lang:
            continue
        if include_langs and lang not in include_langs:
            continue
        if exclude_langs and lang in exclude_langs:
            continue
        pos = str(entry.get("pos") or "").strip()
        gloss = _extract_gloss(entry)
        usage_examples = _extract_usage_examples(entry, limit=max_examples_per_entry)

        lexeme_id = manager.upsert_lexeme(
            lemma=lemma,
            lang=lang,
            part_of_speech=pos,
            gloss=gloss,
            source="wiktextract",
        )

        if materialize_words:
            manager.ensure_word_entry(
                lemma,
                definition=_multilingual_definition(lang=lang, gloss=gloss, part_of_speech=pos),
                part_of_speech=pos,
                usage_examples=usage_examples,
            )

        for translation in _extract_translations(entry, limit=max_translations_per_entry):
            target_lang = str(translation.get("lang") or "").strip()
            target_term = str(translation.get("term") or "").strip()
            if not target_lang or not target_term:
                continue
            manager.db.add_translation(
                lexeme_id=lexeme_id,
                target_lang=target_lang,
                target_term=target_term,
                relation="translation",
                source="wiktextract",
            )
            if target_lang == base_lang:
                manager.db.update_lexeme_base(lemma=lemma, lang=lang, base_term=target_term)
                manager.ensure_base_word(target_term, definition=gloss, part_of_speech=pos)
            elif materialize_words:
                manager.ensure_word_entry(
                    target_term,
                    definition=f"{target_lang} translation linked to '{lemma}'.",
                    usage_examples=[],
                )
            if materialize_words:
                manager.link_word_translation_pair(lemma, target_term)

        count += 1
        if limit and count >= limit:
            break


def ingest_kaikki_jsonl(
    path: str,
    manager: MultilingualManager | None = None,
    base_lang: str = "en",
    limit: int | None = None,
    include_languages: Sequence[str] | str | None = None,
    exclude_languages: Sequence[str] | str | None = None,
    max_examples_per_entry: int = 4,
    max_translations_per_entry: int = 24,
    materialize_words: bool = True,
) -> None:
    """
    Ingest a Kaikki.org JSONL dump.
    """
    manager = manager or MultilingualManager()
    path_obj = Path(path)
    if not path_obj.exists():
        LOGGER.warning("Kaikki JSONL file not found: %s", path)
        return

    include_langs = _normalize_language_filter(include_languages)
    exclude_langs = _normalize_language_filter(exclude_languages)
    count = 0
    for entry in _iter_jsonl(path_obj):
        lemma = str(entry.get("word") or "").strip()
        lang = _entry_language(entry)
        if not lemma or not lang:
            continue
        if include_langs and lang not in include_langs:
            continue
        if exclude_langs and lang in exclude_langs:
            continue
        pos = str(entry.get("pos") or "").strip()
        gloss = _extract_gloss(entry)
        usage_examples = _extract_usage_examples(entry, limit=max_examples_per_entry)

        lexeme_id = manager.upsert_lexeme(
            lemma=lemma,
            lang=lang,
            part_of_speech=pos,
            gloss=gloss,
            source="kaikki",
        )

        if materialize_words:
            manager.ensure_word_entry(
                lemma,
                definition=_multilingual_definition(lang=lang, gloss=gloss, part_of_speech=pos),
                part_of_speech=pos,
                usage_examples=usage_examples,
            )

        for translation in _extract_translations(entry, limit=max_translations_per_entry):
            target_lang = str(translation.get("lang") or "").strip()
            target_term = str(translation.get("term") or "").strip()
            if not target_lang or not target_term:
                continue
            manager.db.add_translation(
                lexeme_id=lexeme_id,
                target_lang=target_lang,
                target_term=target_term,
                relation=str(translation.get("relation") or "translation"),
                source="kaikki",
            )
            if target_lang == base_lang:
                manager.db.update_lexeme_base(lemma=lemma, lang=lang, base_term=target_term)
                manager.ensure_base_word(target_term, definition=gloss, part_of_speech=pos)
            elif materialize_words:
                manager.ensure_word_entry(
                    target_term,
                    definition=f"{target_lang} translation linked to '{lemma}'.",
                    usage_examples=[],
                )
            if materialize_words:
                manager.link_word_translation_pair(lemma, target_term)

        count += 1
        if limit and count >= limit:
            break
