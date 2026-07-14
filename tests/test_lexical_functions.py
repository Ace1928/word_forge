import importlib.util
from pathlib import Path

import pytest

from word_forge.parser.lexical_functions import (
    create_lexical_dataset,
    file_exists,
    get_dbnary_data,
    read_json_file,
    read_jsonl_file,
    safe_open,
)

_RDFLIB_AVAILABLE = importlib.util.find_spec("rdflib") is not None


def test_file_exists(tmp_path: Path) -> None:
    path = tmp_path / "t.txt"
    assert not file_exists(path)
    path.write_text("data")
    assert file_exists(path)


def test_safe_open_missing(tmp_path: Path) -> None:
    with safe_open(tmp_path / "missing.txt") as handle:
        assert handle is None


def test_read_json_file_invalid(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json")
    assert read_json_file(path, {"d": 1}) == {"d": 1}


def test_read_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text('{"a":1}\n{"a":2}\n')

    def proc(data: dict[str, int]) -> int:
        return data["a"]

    assert read_jsonl_file(path, proc) == [1, 2]


def test_read_jsonl_file_missing(tmp_path: Path) -> None:
    path = tmp_path / "missing.jsonl"
    assert read_jsonl_file(path, lambda data: data) == []


def test_lexical_dataset_prefers_source_authored_example(tmp_path: Path) -> None:
    """WordNet examples are preserved without requiring a language model."""
    missing = str(tmp_path / "missing")

    dataset = create_lexical_dataset(
        "dog",
        openthesaurus_path=missing,
        odict_path=missing,
        dbnary_path=missing,
        opendict_path=missing,
        thesaurus_path=missing,
        model_state=None,
    )

    assert dataset["example_sentence"] == "the dog barked all night"


def test_lexical_dataset_leaves_missing_example_empty_without_model(
    tmp_path: Path,
) -> None:
    """The offline path never emits a failure placeholder as lexical data."""
    missing = str(tmp_path / "missing")

    dataset = create_lexical_dataset(
        "oxygen",
        openthesaurus_path=missing,
        odict_path=missing,
        dbnary_path=missing,
        opendict_path=missing,
        thesaurus_path=missing,
        model_state=None,
    )

    assert dataset["wordnet_data"]
    assert dataset["example_sentence"] == ""


@pytest.mark.skipif(not _RDFLIB_AVAILABLE, reason="DBnary RDF extra is not installed")
def test_dbnary_lookup_preserves_definition_and_translation_languages(
    tmp_path: Path,
) -> None:
    rdf_path = tmp_path / "dbnary.ttl"
    rdf_path.write_text(
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

    entries = get_dbnary_data("chat", str(rdf_path), "fr-FR")

    assert entries == [
        {
            "definition": "animal félin",
            "definition_language": "fr",
            "translation": "cat",
            "translation_language": "en",
        }
    ]
