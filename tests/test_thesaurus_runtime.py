from __future__ import annotations

from pathlib import Path

import word_forge.thesaurus_runtime as thesaurus_runtime
from word_forge.database.database_manager import DBManager


class _FakeLemma:
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class _FakeSynset:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def hypernyms(self) -> list["_FakeSynset"]:
        return [_FakeSynset(["collection"])]

    def hyponyms(self) -> list["_FakeSynset"]:
        return [_FakeSynset(["repository"])]

    def lemmas(self) -> list[_FakeLemma]:
        return [_FakeLemma(name) for name in self._names]


def test_run_word_forge_thesaurus_enriches_relationships_and_phonetics(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word("archive", "", "", [])
    db.close()

    monkeypatch.setattr(
        thesaurus_runtime,
        "get_wordnet_data",
        lambda term: [
            {
                "word": term,
                "definition": "A place where records are preserved.",
                "examples": ["The archive preserves institutional memory."],
                "synonyms": ["repository", "records office"],
                "antonyms": ["discard"],
                "part_of_speech": "n",
            }
        ],
    )
    monkeypatch.setattr(thesaurus_runtime, "get_synsets", lambda _term: [_FakeSynset(["archive"])])
    monkeypatch.setattr(
        thesaurus_runtime, "_read_local_thesaurus_synonyms", lambda _term: ["storehouse"]
    )

    class _FakeG2P:
        def convert(self, text: str, context: str | None = None):
            return {"ipa": f"/{text}/", "arpabet": text.upper(), "stress_pattern": "1"}

    monkeypatch.setattr(thesaurus_runtime, "LLMG2PProvider", lambda _model: _FakeG2P())

    result = thesaurus_runtime.run_word_forge_thesaurus(
        repo_root=repo_root,
        db_path=db_path,
        limit=8,
        phonetic_limit=4,
    )

    assert result["status"]["status"] == "completed"
    assert int(result["status"]["relationship_delta"]) >= 4
    assert int(result["status"]["phonetic_delta"]) == 1

    verify = DBManager(db_path=db_path)
    try:
        entry = verify.get_word_entry("archive")
        relationship_terms = {
            (row["relationship_type"], row["related_term"]) for row in entry["relationships"]
        }
        assert ("synonym", "repository") in relationship_terms
        assert ("antonym", "discard") in relationship_terms
        assert verify.word_exists("repository")
    finally:
        verify.close()


def test_select_candidate_terms_prioritizes_missing_relationships(tmp_path: Path) -> None:
    db_path = tmp_path / "word_forge.sqlite"
    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word("archive", "", "", [])
    db.insert_or_update_word("atlas", "A map book", "noun", [])
    db.insert_relationship("atlas", "map", "synonym")
    db.close()

    rows = thesaurus_runtime._select_candidate_terms(db_path, limit=4)
    assert rows
    assert rows[0]["term"] == "archive"


def test_select_candidate_terms_skips_multiword_phrase_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "word_forge.sqlite"
    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word("archive", "", "", [])
    db.insert_or_update_word("middle term", "", "", [])
    db.close()

    rows = thesaurus_runtime._select_candidate_terms(db_path, limit=8)
    terms = [row["term"] for row in rows]
    assert "archive" in terms
    assert "middle term" not in terms


def test_run_word_forge_thesaurus_prioritizes_explicit_seed_terms(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word("archive", "", "", [])
    db.insert_or_update_word("atlas", "", "", [])
    db.insert_or_update_word("provenance", "", "", [])
    db.close()

    monkeypatch.setattr(
        thesaurus_runtime,
        "get_wordnet_data",
        lambda term: [
            {
                "word": term,
                "definition": f"Definition for {term}",
                "examples": [],
                "synonyms": [],
                "antonyms": [],
                "part_of_speech": "n",
            }
        ],
    )
    monkeypatch.setattr(thesaurus_runtime, "get_synsets", lambda _term: [])
    monkeypatch.setattr(thesaurus_runtime, "_read_local_thesaurus_synonyms", lambda _term: [])

    result = thesaurus_runtime.run_word_forge_thesaurus(
        repo_root=repo_root,
        db_path=db_path,
        limit=2,
        phonetic_limit=0,
        seed_terms=["provenance", "atlas"],
        seed_source="test",
        use_llm_phonetics=False,
    )

    assert result["status"]["seed_source"] == "test"
    assert result["status"]["seed_terms"] == ["provenance", "atlas"]
    assert result["report"]["seed_terms"] == ["provenance", "atlas"]
    processed = [row["term"] for row in result["report"]["processed_terms"]]
    assert processed[:2] == ["provenance", "atlas"]


def test_run_word_forge_thesaurus_uses_semantic_neighbors_for_domain_terms(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word("eidosian", "", "", [])
    db.close()

    monkeypatch.setattr(thesaurus_runtime, "get_wordnet_data", lambda term: [])
    monkeypatch.setattr(thesaurus_runtime, "get_synsets", lambda _term: [])
    monkeypatch.setattr(
        thesaurus_runtime,
        "_semantic_related_terms",
        lambda term, limit=4, **kwargs: [
            {
                "term": "workflow",
                "score": 0.81,
                "definition": "flow of work",
                "part_of_speech": "noun",
            },
            {
                "term": "ontology",
                "score": 0.76,
                "definition": "knowledge organization",
                "part_of_speech": "noun",
            },
        ],
    )

    result = thesaurus_runtime.run_word_forge_thesaurus(
        repo_root=repo_root,
        db_path=db_path,
        limit=4,
        phonetic_limit=0,
        seed_terms=["eidosian"],
        use_llm_phonetics=False,
    )

    assert int(result["status"]["semantic_related_delta"]) >= 2
    assert int(result["status"]["semantic_neighbor_hit_count"]) >= 1

    verify = DBManager(db_path=db_path)
    try:
        entry = verify.get_word_entry("eidosian")
        relationship_terms = {
            (row["relationship_type"], row["related_term"]) for row in entry["relationships"]
        }
        assert ("related", "workflow") in relationship_terms
        assert ("related", "ontology") in relationship_terms
    finally:
        verify.close()


def test_run_word_forge_thesaurus_adds_omw_translations(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word("hello", "", "", [])
    db.close()

    monkeypatch.setattr(thesaurus_runtime, "get_wordnet_data", lambda term: [])
    monkeypatch.setattr(thesaurus_runtime, "get_synsets", lambda _term: [])
    monkeypatch.setattr(thesaurus_runtime, "_semantic_related_terms", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        thesaurus_runtime,
        "_omw_translation_terms",
        lambda term, limit_per_language=4: {"spa": ["hola"], "fra": ["bonjour"]},
    )

    result = thesaurus_runtime.run_word_forge_thesaurus(
        repo_root=repo_root,
        db_path=db_path,
        limit=4,
        phonetic_limit=0,
        seed_terms=["hello"],
        use_llm_phonetics=False,
    )

    assert int(result["status"]["omw_translation_delta"]) >= 2
    assert int(result["status"]["omw_language_hit_count"]) == 2

    verify = DBManager(db_path=db_path)
    try:
        entry = verify.get_word_entry("hello")
        relationship_terms = {
            (row["relationship_type"], row["related_term"]) for row in entry["relationships"]
        }
        assert ("translation", "hola") in relationship_terms
        assert ("translation", "bonjour") in relationship_terms
        assert verify.word_exists("hola")
        assert verify.word_exists("bonjour")
    finally:
        verify.close()


def test_aggregate_wordnet_relationships_includes_derivational_forms(monkeypatch) -> None:
    class _FakeLemma:
        def __init__(self, name: str, related: list[str] | None = None):
            self._name = name
            self._related = related or []

        def name(self):
            return self._name

        def antonyms(self):
            return []

        def derivationally_related_forms(self):
            return [_FakeLemma(item) for item in self._related]

    class _FakeSynset:
        def lemmas(self):
            return [
                _FakeLemma("integrate", ["integration"]),
                _FakeLemma("integration", ["integrate"]),
            ]

        def hypernyms(self):
            return []

        def hyponyms(self):
            return []

        def definition(self):
            return "to combine parts into a whole"

        def examples(self):
            return []

        def pos(self):
            return "v"

    monkeypatch.setattr(thesaurus_runtime, "get_synsets", lambda _term: [_FakeSynset()])
    monkeypatch.setattr(
        thesaurus_runtime,
        "get_wordnet_data",
        lambda term: [
            {
                "word": term,
                "definition": "to combine parts into a whole",
                "examples": [],
                "synonyms": [term],
                "antonyms": [],
                "part_of_speech": "v",
            }
        ],
    )

    payload = thesaurus_runtime._aggregate_wordnet_relationships("integrate")
    relationships = payload["relationships"]
    assert (
        "integration" in relationships["etymological_source"]
        or "integration" in relationships["derived_from"]
    )
