"""Behavioral tests for the friendly high-level Lexicon API."""

from pathlib import Path

import pytest

from word_forge import Lexicon
from word_forge.database.database_manager import RevisionConflictError, TermNotFoundError


def test_add_get_find_and_statistics(tmp_path: Path) -> None:
    with Lexicon(tmp_path / "lexicon.sqlite") as lexicon:
        created = lexicon.add(
            "recursion",
            definition="A process that refers to itself.",
            part_of_speech="noun",
            examples=["Recursion can traverse a tree.", "Recursion can traverse a tree."],
        )

        assert created.term == "recursion"
        assert len(created.id) == 36
        assert created.normalized_term == "recursion"
        assert created.script == "Latn"
        assert created.revision == 1
        assert created.examples == ("Recursion can traverse a tree.",)
        assert lexicon.get("recursion").definition.startswith("A process")
        assert [entry.term for entry in lexicon.find("curs")] == ["recursion"]
        assert lexicon.statistics().as_dict() == {
            "words": 1,
            "lexemes": 1,
            "languages": 1,
            "relationships": 0,
            "translations": 0,
        }


def test_multilingual_entries_and_bidirectional_translation(tmp_path: Path) -> None:
    with Lexicon(tmp_path / "lexicon.sqlite") as lexicon:
        lexicon.add("house", definition="A building used as a home.")
        lexicon.add(
            "casa",
            language="es",
            definition="Edificio para habitar.",
            base_term="house",
        )
        lexicon.translate(
            "house",
            source_language="en",
            target_language="de",
            target_term="Haus",
        )

        spanish = lexicon.get("casa", language="es")
        german = lexicon.get("Haus", language="de")
        assert spanish.base_term == "house"
        assert ("house", "en") in {
            (relationship.target, relationship.language) for relationship in spanish.relationships
        }
        assert ("house", "en") in {
            (relationship.target, relationship.language) for relationship in german.relationships
        }
        assert [entry.term for entry in lexicon.find("edificio", language="es")] == ["casa"]


def test_relationship_creation_ensures_target_exists(tmp_path: Path) -> None:
    with Lexicon(tmp_path / "lexicon.sqlite") as lexicon:
        lexicon.add("algorithm")
        assert lexicon.relate("algorithm", "procedure", kind="synonym") is True
        relationship = lexicon.get("algorithm").relationships[0]
        assert (relationship.target, relationship.kind) == ("procedure", "synonym")


def test_invalid_input_and_closed_state_are_actionable(tmp_path: Path) -> None:
    lexicon = Lexicon(tmp_path / "lexicon.sqlite")
    with pytest.raises(ValueError, match="term must not be empty"):
        lexicon.add("   ")
    with pytest.raises(ValueError, match="language must be a tag"):
        lexicon.add("hello", language="not a language")
    with pytest.raises(TermNotFoundError):
        lexicon.get("missing")

    lexicon.close()
    lexicon.close()
    with pytest.raises(RuntimeError, match="closed"):
        lexicon.statistics()


def test_like_metacharacters_are_treated_as_text(tmp_path: Path) -> None:
    with Lexicon(tmp_path / "lexicon.sqlite") as lexicon:
        lexicon.add("100%")
        lexicon.add("1000")
        assert [entry.term for entry in lexicon.find("%", language="en")] == ["100%"]


def test_revisions_prevent_stale_overwrites(tmp_path: Path) -> None:
    with Lexicon(tmp_path / "lexicon.sqlite") as lexicon:
        created = lexicon.add("state", definition="Initial", expected_revision=0)
        updated = lexicon.add(
            "state",
            definition="Current",
            expected_revision=created.revision,
        )
        assert updated.id == created.id
        assert updated.revision == created.revision + 1
        with pytest.raises(RevisionConflictError, match="Revision conflict"):
            lexicon.add("state", definition="Stale", expected_revision=created.revision)
        assert lexicon.get("state").definition == "Current"


def test_senses_are_independently_versioned_and_tombstoned(tmp_path: Path) -> None:
    with Lexicon(tmp_path / "lexicon.sqlite") as lexicon:
        lexicon.add("bank")
        financial = lexicon.add_sense(
            "bank",
            "A financial institution.",
            domain="finance",
            part_of_speech="noun",
            source="editorial",
            expected_revision=0,
        )
        river = lexicon.add_sense(
            "bank",
            "The land beside a river.",
            domain="geography",
        )
        revised = lexicon.add_sense(
            "bank",
            "An institution that holds and lends money.",
            domain="finance",
            part_of_speech="noun",
            source="editorial",
            sense_id=financial.id,
            expected_revision=financial.revision,
        )
        assert revised.revision == 2
        assert {sense.domain for sense in lexicon.get("bank").senses} == {
            "finance",
            "geography",
        }
        assert lexicon.remove_sense(river.id, expected_revision=river.revision) is True
        assert [sense.domain for sense in lexicon.get("bank").senses] == ["finance"]
