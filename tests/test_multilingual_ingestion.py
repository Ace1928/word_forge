"""Real pipeline tests for language-aware lexical-form persistence."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from word_forge.configs.config_essentials import LexicalDataset
from word_forge.database.database_manager import DBManager
from word_forge.parser.parser_refiner import ParserRefiner
from word_forge.queue.queue_manager import QueueManager
from word_forge.utils.nltk_utils import ensure_nltk_data

_RDFLIB_AVAILABLE = importlib.util.find_spec("rdflib") is not None


def test_non_english_seed_persists_language_and_graphemes_without_omw(
    tmp_path: Path,
) -> None:
    ensure_nltk_data()
    database = DBManager(db_path=tmp_path / "french.sqlite")
    queue: QueueManager[str] = QueueManager()
    queue.start()
    parser = ParserRefiner(
        db_manager=database,
        queue_manager=queue,
        language="fr-FR",
    )

    try:
        assert parser.process_word("café") is True
        entry = database.get_word_entry("CAFÉ", "fr-FR")

        assert entry["language"] == "fr-FR"
        assert entry["script"] == "Latn"
        assert [grapheme["text"] for grapheme in entry["graphemes"]] == list("café")
        # CMUdict is English-only; no pronunciation coverage is fabricated.
        assert entry["pronunciations"] == []
    finally:
        parser.shutdown()
        queue.stop()
        database.close()


def test_non_english_ingestion_ignores_unlabelled_legacy_english_files(
    tmp_path: Path,
) -> None:
    """Legacy local formats cannot be relabelled as another language."""

    (tmp_path / "odict.json").write_text(
        '{"chat": {"definition": "an informal conversation", "examples": []}}',
        encoding="utf-8",
    )
    (tmp_path / "opendict.json").write_text(
        '{"chat": {"definition": "talk in a friendly way", "examples": []}}',
        encoding="utf-8",
    )
    (tmp_path / "openthesaurus.jsonl").write_text(
        '{"words": ["chat", "talk"]}\n',
        encoding="utf-8",
    )
    (tmp_path / "thesaurus.jsonl").write_text(
        '{"word": "chat", "synonyms": ["conversation"]}\n',
        encoding="utf-8",
    )
    database = DBManager(db_path=tmp_path / "language-boundary.sqlite")
    queue: QueueManager[str] = QueueManager()
    queue.start()
    parser = ParserRefiner(
        db_manager=database,
        queue_manager=queue,
        data_dir=str(tmp_path),
        language="fr",
    )

    try:
        assert parser.process_word("chat") is True
        entry = database.get_word_entry("chat", "fr")

        assert entry["definition"] == ""
        assert not {
            relationship["related_term"] for relationship in entry["relationships"]
        }.intersection({"talk", "conversation"})
    finally:
        parser.shutdown()
        queue.stop()
        database.close()


def test_definition_extraction_keeps_only_the_requested_language() -> None:
    dataset: LexicalDataset = {
        "word": "chat",
        "language": "fr-FR",
        "wordnet_data": [
            {
                "word": "chat",
                "language": "fr-FR",
                "source": "open-multilingual-wordnet",
                "synset_id": "cat.n.01",
                "definition": "feline mammal usually having thick soft fur",
                "definition_language": "en",
                "examples": [],
                "examples_language": "en",
                "synonyms": ["chat"],
                "antonyms": [],
                "part_of_speech": "n",
            }
        ],
        "openthesaurus_synonyms": [],
        "odict_data": {"definition": "Not Found", "examples": []},
        "dbnary_data": [
            {
                "definition": "cat-like mammal",
                "definition_language": "en",
                "translation": "",
                "translation_language": "",
            },
            {
                "definition": "mammifère carnivore de la famille des félidés",
                "definition_language": "fr",
                "translation": "",
                "translation_language": "",
            },
        ],
        "opendict_data": {"definition": "Not Found", "examples": []},
        "thesaurus_synonyms": [],
        "example_sentence": "",
        "source_warnings": [],
    }

    assert ParserRefiner._extract_all_definitions(dataset) == [
        "mammifère carnivore de la famille des félidés"
    ]


def test_english_ingestion_persists_source_backed_pronunciations(
    tmp_path: Path,
) -> None:
    ensure_nltk_data()
    database = DBManager(db_path=tmp_path / "english.sqlite")
    queue: QueueManager[str] = QueueManager()
    queue.start()
    parser = ParserRefiner(
        db_manager=database,
        queue_manager=queue,
        language="en-US",
    )

    try:
        assert parser.process_word("Cat") is True
        entry = database.get_word_entry("cat", "en-US")

        assert entry["term"] == "Cat"
        assert entry["source"] == "princeton-wordnet"
        assert {item["notation"] for item in entry["pronunciations"]} == {
            "arpabet",
            "ipa",
        }
        assert all(
            item["source"] in {"cmudict", "cmudict-derived"}
            for item in entry["pronunciations"]
        )
        assert entry["relationships"]
        assert all(
            relationship["related_language"] == "en-US"
            for relationship in entry["relationships"]
        )
    finally:
        parser.shutdown()
        queue.stop()
        database.close()


@pytest.mark.skipif(not _RDFLIB_AVAILABLE, reason="DBnary RDF extra is not installed")
def test_cross_language_translation_is_tagged_without_wrong_queue_expansion(
    tmp_path: Path,
) -> None:
    (tmp_path / "dbnary.ttl").write_text(
        """
        @prefix ontolex: <http://www.w3.org/ns/lemon/ontolex#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix ex: <https://example.test/> .

        ex:entry
            ontolex:canonicalForm [ ontolex:writtenRep "chat"@fr ] ;
            ontolex:definition [ rdfs:label "animal félin"@fr ] ;
            ontolex:translation [ rdfs:label "cat"@en ] .
        """,
        encoding="utf-8",
    )
    database = DBManager(db_path=tmp_path / "translations.sqlite")
    queue: QueueManager[str] = QueueManager()
    queue.start()
    parser = ParserRefiner(
        db_manager=database,
        queue_manager=queue,
        data_dir=str(tmp_path),
        language="fr",
    )

    try:
        assert parser.process_word("chat") is True
        entry = database.get_word_entry("chat", "fr")
        translations = [
            relation
            for relation in entry["relationships"]
            if relation["relationship_type"] == "translation"
        ]

        assert translations == [
            {
                "related_term": "cat",
                "related_normalized_term": "cat",
                "related_language": "en",
                "relationship_type": "translation",
                "source": "dbnary",
                "confidence": 1.0,
            }
        ]
        assert "cat" not in queue.seen_items()
    finally:
        parser.shutdown()
        queue.stop()
        database.close()
