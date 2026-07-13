from __future__ import annotations

import sqlite3
from pathlib import Path

import word_forge.growth_runtime as growth_runtime
from word_forge.database.database_manager import DBManager


def test_rank_forge_local_candidates_prefers_cross_forge_support() -> None:
    rows = growth_runtime._rank_forge_local_candidates(
        existing_terms={"hello"},
        knowledge_tag_tokens={"archive", "atlas", "hello"},
        file_path_tokens={"archive", "integration", "atlas"},
        code_tokens={"archive", "integration", "pipeline"},
        morpheme_anchor_terms={"archive", "horizon"},
        repo_domain_tokens={"atlas", "integration", "semantic"},
        docs_tokens={"ontology", "semantic", "integration"},
        prompt_tokens={"workflow", "semantic"},
        proof_tokens={"provenance", "atlas"},
        limit=8,
    )

    terms = [row["term"] for row in rows]
    assert "archive" in terms
    assert "atlas" in terms
    assert "integration" in terms
    assert "hello" not in terms


def test_rank_forge_local_candidates_keeps_supported_generic_terms() -> None:
    rows = growth_runtime._rank_forge_local_candidates(
        existing_terms=set(),
        knowledge_tag_tokens={"archive", "while", "project"},
        file_path_tokens={"archive", "while", "project"},
        code_tokens={"archive", "while", "project"},
        morpheme_anchor_terms={"archive"},
        repo_domain_tokens={"archive", "while", "project"},
        docs_tokens={"archive", "while", "project"},
        prompt_tokens={"archive", "while", "project"},
        proof_tokens={"archive", "while", "project"},
        limit=8,
    )

    terms = [row["term"] for row in rows]
    assert "archive" in terms
    assert "while" in terms
    assert "project" in terms


def test_resolve_growth_phonetic_limit_adapts_to_coverage_gap() -> None:
    low_coverage = growth_runtime._resolve_growth_phonetic_limit(
        {"word_count": 1000, "phonetic_word_count": 5},
        requested_limit=None,
        thesaurus_limit=24,
    )
    medium_coverage = growth_runtime._resolve_growth_phonetic_limit(
        {"word_count": 1000, "phonetic_word_count": 120},
        requested_limit=None,
        thesaurus_limit=24,
    )
    requested = growth_runtime._resolve_growth_phonetic_limit(
        {"word_count": 1000, "phonetic_word_count": 5},
        requested_limit=3,
        thesaurus_limit=24,
    )

    assert low_coverage == 16
    assert medium_coverage == 8
    assert requested == 3


def test_resolve_growth_emotion_limit_adapts_to_coverage_gap() -> None:
    low_coverage = growth_runtime._resolve_growth_emotion_limit(
        {"word_count": 1000, "emotion_word_count": 0},
        requested_limit=None,
        thesaurus_limit=24,
    )
    medium_coverage = growth_runtime._resolve_growth_emotion_limit(
        {"word_count": 1000, "emotion_word_count": 120},
        requested_limit=None,
        thesaurus_limit=24,
    )

    assert low_coverage == 16
    assert medium_coverage == 8


def test_resolve_growth_derivation_limit_adapts_to_coverage_gap() -> None:
    low_coverage = growth_runtime._resolve_growth_derivation_limit(
        {"word_count": 1000, "derivation_word_count": 5},
        requested_limit=None,
        thesaurus_limit=24,
    )
    medium_coverage = growth_runtime._resolve_growth_derivation_limit(
        {"word_count": 1000, "derivation_word_count": 150},
        requested_limit=None,
        thesaurus_limit=24,
    )

    assert low_coverage == 24
    assert medium_coverage == 12


def test_resolve_growth_focus_ranks_weakest_dimensions_first() -> None:
    focus = growth_runtime._resolve_growth_focus(
        {
            "word_count": 1000,
            "defined_word_count": 800,
            "pos_word_count": 700,
            "usage_example_word_count": 600,
            "phonetic_word_count": 40,
            "emotion_word_count": 10,
            "derivation_word_count": 25,
            "translation_word_count": 200,
        }
    )
    assert focus[0]["dimension"] == "emotion"
    assert focus[1]["dimension"] == "derivation"
    assert focus[0]["recommended_family"] == "emotion"


def test_harvest_derivation_seed_rows_returns_defined_terms_without_derivation(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "word_forge.sqlite"
    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word("archive", "stored records and documents", "noun", ["archive example"])
    db.insert_or_update_word("archival", "related to archives", "adjective", ["archival example"])
    db.insert_relationship("archival", "archive", "derived_from")
    db.close()

    rows = growth_runtime._harvest_derivation_seed_rows(db_path, limit=8)
    terms = [row["term"] for row in rows]
    assert "archive" in terms
    assert "archival" not in terms


def test_harvest_usage_seed_rows_discovers_missing_terms_from_examples(tmp_path: Path) -> None:
    db_path = tmp_path / "word_forge.sqlite"
    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word(
        "archive",
        "stored records",
        "noun",
        ["The archive supports provenance and ontology mapping."],
    )
    db.close()

    rows = growth_runtime._harvest_usage_seed_rows(db_path, limit=12)
    terms = [row["term"] for row in rows]
    assert "provenance" in terms
    assert "ontology" in terms


def test_run_forge_local_discovery_inserts_missing_terms(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word("hello", "A greeting", "noun", ["hello world"])
    db.close()

    monkeypatch.setattr(growth_runtime, "_load_word_metrics", lambda _: {"word_terms": {"hello"}})
    monkeypatch.setattr(
        growth_runtime,
        "_load_knowledge_tokens",
        lambda _: {
            "tag_tokens": {"archive", "atlas", "integration"},
            "content_tokens": {"ignoreme"},
        },
    )
    monkeypatch.setattr(
        growth_runtime,
        "_load_file_tokens",
        lambda _: {
            "path_tokens": {"archive", "atlas", "integration"},
            "semantic_tokens": {"pipeline"},
        },
    )
    monkeypatch.setattr(
        growth_runtime,
        "_load_code_tokens",
        lambda *_args, **_kwargs: {"code_tokens": {"archive", "integration", "pipeline"}},
    )
    monkeypatch.setattr(
        growth_runtime,
        "_load_morpheme_metrics",
        lambda _: {"informative_anchor_morphemes": {"archive": ["arch", "ive"]}},
    )
    monkeypatch.setattr(
        growth_runtime,
        "_discover_repo_domain_tokens",
        lambda _: {"atlas", "integration", "semantic"},
    )
    monkeypatch.setattr(
        growth_runtime,
        "_load_text_corpus_tokens",
        lambda paths, **kwargs: (
            {"ontology", "workflow", "semantic"}
            if any("docs" in str(path) for path in paths)
            else (
                {"prompt", "workflow"}
                if any("prompt" in str(path) for path in paths)
                else {"proof", "provenance"}
            )
        ),
    )

    result = growth_runtime.run_forge_local_discovery(repo_root=repo_root, db_path=db_path, limit=8)

    assert result["status"] == "completed"
    assert result["inserted_count"] >= 3
    inserted = set(result["inserted_terms"])
    assert {"archive", "atlas", "integration"}.issubset(inserted)

    verify = DBManager(db_path=db_path)
    try:
        for term in ("archive", "atlas", "integration"):
            assert verify.get_word_id(term) > 0
    finally:
        verify.close()


def test_discover_growth_sources_reports_corpus_local_count(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "word_forge" / "data").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "word_forge").mkdir(parents=True, exist_ok=True)
    (repo_root / "data" / "kb.json").write_text("{}", encoding="utf-8")
    (repo_root / "docs").mkdir(parents=True, exist_ok=True)
    (repo_root / "prompts").mkdir(parents=True, exist_ok=True)
    (repo_root / "reports" / "proof").mkdir(parents=True, exist_ok=True)
    (repo_root / "reports" / "word_forge_bridge_audit").mkdir(parents=True, exist_ok=True)
    (repo_root / "reports" / "proof" / "entity_proof_scorecard_latest.json").write_text(
        "{}", encoding="utf-8"
    )
    (repo_root / "reports" / "word_forge_bridge_audit" / "latest.json").write_text(
        "{}", encoding="utf-8"
    )

    payload = growth_runtime.discover_growth_sources(repo_root)
    assert int(payload["forge_local_source_count"]) >= 4
    assert int(payload["corpus_local_source_count"]) >= 3


def test_cleanup_forge_local_discovery_terms_removes_words_and_vectors(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    vec_dir = repo_root / "word_forge" / "data" / "vector_indices"
    vec_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word(
        "archive",
        f"{growth_runtime._FORGE_LOCAL_DISCOVERY_DEFINITION_PREFIX}knowledge, code substrates of the Eidosian system for the term 'archive'.",
        "noun",
        ["archive example"],
    )
    word_id = db.get_word_id("archive")
    db.close()

    import sqlite3

    vec_db = vec_dir / "vector_store.sqlite3"
    with sqlite3.connect(vec_db) as conn:
        conn.execute(
            "CREATE TABLE vectors (id TEXT PRIMARY KEY, embedding BLOB NOT NULL, metadata TEXT, document TEXT)"
        )
        conn.execute("CREATE TABLE collection_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "INSERT INTO vectors(id, embedding, metadata, document) VALUES (?, ?, ?, ?)",
            (
                "w_1",
                sqlite3.Binary(b"0" * 16),
                '{"original_id": %d, "term": "archive"}' % word_id,
                "archive",
            ),
        )
        conn.commit()
    (vec_dir / "vector_store.hnsw.bin").write_bytes(b"placeholder")

    result = growth_runtime.cleanup_forge_local_discovery_terms(
        repo_root=repo_root, db_path=db_path
    )

    assert result["deleted_word_count"] == 1
    assert result["deleted_vector_count"] == 1

    verify = DBManager(db_path=db_path)
    try:
        try:
            verify.get_word_id("archive")
            raise AssertionError("archive should have been removed")
        except Exception:
            pass
    finally:
        verify.close()


def test_run_word_forge_growth_daemon_completes_one_cycle(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    vector_db_path = repo_root / "data" / "word_forge_fasttext.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    vector_db_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        growth_runtime,
        "run_word_forge_growth",
        lambda **_kwargs: {
            "status": {"status": "completed", "finished_at": "2026-04-01T00:00:00+00:00"}
        },
    )

    result = growth_runtime.run_word_forge_growth_daemon(
        repo_root=repo_root,
        db_path=db_path,
        vector_db_path=vector_db_path,
        interval_seconds=30,
        cycles=1,
    )

    assert result["status"] == "completed"
    assert result["cycle_count"] == 1


def test_run_word_forge_growth_daemon_passes_thesaurus_limits(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    vector_db_path = repo_root / "data" / "word_forge_fasttext.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    vector_db_path.parent.mkdir(parents=True, exist_ok=True)
    captured: dict[str, object] = {}

    def _fake_growth(**kwargs):
        captured.update(kwargs)
        return {"status": {"status": "completed", "finished_at": "2026-04-01T00:00:00+00:00"}}

    monkeypatch.setattr(growth_runtime, "run_word_forge_growth", _fake_growth)

    growth_runtime.run_word_forge_growth_daemon(
        repo_root=repo_root,
        db_path=db_path,
        vector_db_path=vector_db_path,
        interval_seconds=30,
        cycles=1,
        thesaurus_limit=12,
        phonetic_limit=3,
    )

    assert captured["thesaurus_limit"] == 12
    assert captured["phonetic_limit"] == 3


def test_request_word_forge_growth_daemon_stop_writes_signal(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    result = growth_runtime.request_word_forge_growth_daemon_stop(repo_root=repo_root)
    assert result["status"] == "requested"
    assert (repo_root / "data" / "runtime" / growth_runtime.DAEMON_STOP_FILE_NAME).exists()


def test_run_word_forge_growth_daemon_passes_thesaurus_terms(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    vector_db_path = repo_root / "data" / "word_forge_fasttext.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    vector_db_path.parent.mkdir(parents=True, exist_ok=True)
    captured: dict[str, object] = {}

    def fake_run_word_forge_growth(**kwargs):
        captured.update(kwargs)
        return {
            "status": {"status": "completed", "phase": "completed"},
            "report": {},
            "artifacts": {},
        }

    monkeypatch.setattr(
        "word_forge.growth_runtime.run_word_forge_growth", fake_run_word_forge_growth
    )
    from word_forge.growth_runtime import run_word_forge_growth_daemon

    result = run_word_forge_growth_daemon(
        repo_root=repo_root,
        db_path=db_path,
        vector_db_path=vector_db_path,
        interval_seconds=30,
        cycles=1,
        force_first_cycle=True,
        thesaurus_terms=["archive", "atlas"],
    )

    assert result["status"] == "completed"
    assert captured["thesaurus_terms"] == ["archive", "atlas"]


def test_run_lexicon_backfill_populates_words_translations_and_morphemes(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = DBManager(db_path=db_path)
    db.create_tables()
    lexeme_id = db.insert_or_update_lexeme(
        lemma="hola",
        lang="es",
        part_of_speech="interjection",
        gloss="A greeting used to say hello.",
        base_term="hello",
        source="test",
    )
    db.add_translation(
        lexeme_id, target_lang="en", target_term="hello", relation="translation", source="test"
    )
    db.insert_or_update_lexeme(
        lemma="embedding",
        lang="en",
        part_of_speech="noun",
        gloss="The act of placing something firmly within another thing.",
        base_term="embedding",
        source="test",
    )
    db.close()

    result = growth_runtime.run_lexicon_backfill(repo_root=repo_root, db_path=db_path, limit=12)

    assert result["status"] == "completed"
    assert result["inserted_word_count"] >= 2
    assert result["translation_relationship_delta"] >= 2
    assert result["morpheme_link_delta"] >= 1
    assert "es" in result["languages"]

    verify = DBManager(db_path=db_path)
    try:
        assert verify.get_word_id("hello") > 0
        assert verify.get_word_id("hola") > 0
        assert verify.get_word_id("embedding") > 0
        relationships = verify.execute_query(
            "SELECT r.related_term, r.relationship_type FROM relationships r JOIN words w ON w.id = r.word_id WHERE w.term = ?",
            ("hello",),
        )
        assert any(
            str(row["related_term"]) == "hola" and str(row["relationship_type"]) == "translation"
            for row in relationships
        )
        morphemes = verify.execute_query(
            "SELECT m.text FROM morphemes m JOIN word_morphemes wm ON wm.morpheme_id = m.id JOIN words w ON w.id = wm.word_id WHERE w.term = ? ORDER BY wm.position",
            ("embedding",),
        )
        assert [str(row["text"]) for row in morphemes]
    finally:
        verify.close()


def test_run_word_forge_growth_daemon_passes_lexicon_limit(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    vector_db_path = repo_root / "data" / "word_forge_fasttext.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    vector_db_path.parent.mkdir(parents=True, exist_ok=True)
    captured: dict[str, object] = {}

    def _fake_growth(**kwargs):
        captured.update(kwargs)
        return {"status": {"status": "completed", "finished_at": "2026-04-01T00:00:00+00:00"}}

    monkeypatch.setattr(growth_runtime, "run_word_forge_growth", _fake_growth)

    growth_runtime.run_word_forge_growth_daemon(
        repo_root=repo_root,
        db_path=db_path,
        vector_db_path=vector_db_path,
        interval_seconds=30,
        cycles=1,
        lexicon_limit=88,
        target_word_count=10000,
    )

    assert captured["lexicon_limit"] == 88


def test_synthesise_growth_thesaurus_terms_merges_growth_sources() -> None:
    terms = growth_runtime._synthesise_growth_thesaurus_terms(
        explicit_terms=["archive", "Atlas", ""],
        forge_local_run={
            "inserted_terms": ["eidosian", "workflow"],
            "candidates": [{"term": "ontology"}, {"term": "archive"}, {"term": "graph"}],
        },
        lexicon_backfill_run={"terms": ["hola", "bonjour", "ontology"]},
        multilingual_seed_terms=["salut", "hola"],
        definition_seed_terms=["grammar"],
        limit=8,
    )

    assert terms == [
        "archive",
        "atlas",
        "eidosian",
        "workflow",
        "ontology",
        "graph",
        "hola",
        "bonjour",
    ]


def test_harvest_multilingual_seed_terms_prefers_translation_dense_words(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word("hello", "greeting", "noun", [])
    db.insert_or_update_word("hola", "saludo", "noun", [])
    db.insert_or_update_word("bonjour", "salut", "noun", [])
    hello_id = db.get_word_id("hello")
    hola_id = db.get_word_id("hola")
    bonjour_id = db.get_word_id("bonjour")
    db.execute_query(
        "INSERT INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, ?)",
        (hello_id, "hola", "translation"),
    )
    db.execute_query(
        "INSERT INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, ?)",
        (hello_id, "bonjour", "translation"),
    )
    db.execute_query(
        "INSERT INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, ?)",
        (hola_id, "hello", "translation"),
    )
    db.execute_query(
        "INSERT INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, ?)",
        (bonjour_id, "hello", "translation"),
    )
    db.insert_or_update_lexeme(
        lemma="hola",
        lang="es",
        part_of_speech="interjection",
        gloss="hello",
        base_term="hello",
        source="test",
    )
    db.insert_or_update_lexeme(
        lemma="bonjour",
        lang="fr",
        part_of_speech="interjection",
        gloss="hello",
        base_term="hello",
        source="test",
    )
    db.close()

    terms = growth_runtime._harvest_multilingual_seed_terms(db_path, limit=8)
    assert "hello" in terms
    assert "hola" in terms or "bonjour" in terms


def test_harvest_multilingual_seed_rows_include_signal_counts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word("hello", "greeting", "noun", [])
    hello_id = db.get_word_id("hello")
    db.execute_query(
        "INSERT INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, ?)",
        (hello_id, "hola", "translation"),
    )
    db.execute_query(
        "INSERT INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, ?)",
        (hello_id, "bonjour", "translation"),
    )
    db.insert_or_update_lexeme(
        lemma="hola",
        lang="es",
        part_of_speech="interjection",
        gloss="hello",
        base_term="hello",
        source="test",
    )
    db.insert_or_update_lexeme(
        lemma="bonjour",
        lang="fr",
        part_of_speech="interjection",
        gloss="hello",
        base_term="hello",
        source="test",
    )
    db.close()

    rows = growth_runtime._harvest_multilingual_seed_rows(db_path, limit=4)
    assert rows[0]["term"] == "hello"
    assert rows[0]["translation_links"] >= 2
    assert rows[0]["language_count"] >= 2


def test_build_growth_priority_queue_merges_growth_sources() -> None:
    rows = growth_runtime._build_growth_priority_queue(
        forge_local_run={
            "candidates": [
                {
                    "term": "archive",
                    "score": 31,
                    "support_count": 3,
                    "sources": ["code", "docs", "proof"],
                },
                {"term": "hola", "score": 12, "support_count": 2, "sources": ["docs", "repo"]},
            ]
        },
        lexicon_backfill_run={"terms": ["hola", "bonjour"]},
        multilingual_seed_rows=[
            {"term": "hola", "translation_links": 3, "language_count": 2},
            {"term": "bonjour", "translation_links": 2, "language_count": 2},
        ],
        incomplete_seed_rows=[
            {
                "term": "hola",
                "missing_fields": ["phonetics"],
                "relationship_count": 1,
                "translation_count": 3,
            }
        ],
        definition_seed_rows=[
            {
                "term": "grammar",
                "mention_count": 3,
                "source_term_count": 2,
                "is_function_word": False,
            }
        ],
        limit=8,
    )
    assert rows[0]["term"] == "hola"
    assert "multilingual" in rows[0]["source_families"]
    assert "lexicon_backfill" in rows[0]["source_families"]
    assert rows[0]["priority_score"] > rows[1]["priority_score"]


def test_build_growth_priority_queue_includes_vector_guidance_rows() -> None:
    rows = growth_runtime._build_growth_priority_queue(
        forge_local_run={},
        lexicon_backfill_run={},
        multilingual_seed_rows=[],
        incomplete_seed_rows=[],
        definition_seed_rows=[],
        usage_seed_rows=[],
        vector_seed_rows=[
            {
                "term": "cometido",
                "source_term_count": 2,
                "semantic_neighbor_count": 2,
                "max_score": 0.77,
                "missing_term": True,
            },
        ],
        coverage_focus=[{"dimension": "emotion", "preferred_mode": "semantic", "gap": 0.9}],
        limit=8,
    )
    assert rows[0]["term"] == "cometido"
    assert "vector_guidance" in rows[0]["source_families"]
    assert any("semantic" in reason for reason in rows[0]["reasons"])


def test_harvest_vector_seed_rows_uses_broader_weak_term_pool(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word("project", "a planned undertaking", "noun", ["project planning"])
    db.close()

    monkeypatch.setattr(
        growth_runtime, "_harvest_incomplete_word_rows", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        growth_runtime,
        "_harvest_definition_seed_rows",
        lambda *_args, **_kwargs: [{"term": "project"}],
    )
    monkeypatch.setattr(growth_runtime, "_harvest_usage_seed_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(growth_runtime, "_harvest_emotion_seed_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        growth_runtime, "_harvest_derivation_seed_rows", lambda *_args, **_kwargs: []
    )

    class _FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, **kwargs):
            return [{"distance": 0.2, "metadata": {"term": "contener", "content_type": "word"}}]

    monkeypatch.setitem(growth_runtime.__dict__, "VectorStore", _FakeStore)

    class _FakeStorageType:
        DISK = "disk"

    monkeypatch.setitem(growth_runtime.__dict__, "StorageType", _FakeStorageType)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "word_forge.configs.config_essentials":
            mod = type("M", (), {"StorageType": _FakeStorageType})
            return mod
        if name == "word_forge.vectorizer.vector_store":
            mod = type("M", (), {"VectorStore": _FakeStore})
            return mod
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    rows = growth_runtime._harvest_vector_seed_rows(repo_root, db_path, limit=8)
    assert rows
    assert rows[0]["term"] == "contener"
    assert rows[0]["missing_part_of_speech"] is True
    assert rows[0]["placeholder_definition"] is True
    assert rows[0]["translation_thin"] is False


def test_harvest_vector_seed_rows_prioritises_translation_thin_multilingual_terms(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word("project", "a planned undertaking", "noun", ["project planning"])
    db.insert_or_update_word("hola", "A spanish greeting.", "", [])
    hello_id = db.get_word_id("hola")
    db.execute_query(
        "INSERT INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, ?)",
        (hello_id, "hello", "translation"),
    )
    db.close()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "INSERT INTO lexemes (lemma, lang, source, base_term, last_refreshed) VALUES (?, ?, ?, ?, datetime('now'))",
            ("hola", "es", "seed", "hola"),
        )
        conn.execute(
            "INSERT INTO lexemes (lemma, lang, source, base_term, last_refreshed) VALUES (?, ?, ?, ?, datetime('now'))",
            ("hello", "en", "seed", "hola"),
        )
        conn.execute(
            "INSERT INTO lexemes (lemma, lang, source, base_term, last_refreshed) VALUES (?, ?, ?, ?, datetime('now'))",
            ("bonjour", "fr", "seed", "hola"),
        )
        conn.commit()

    monkeypatch.setattr(
        growth_runtime,
        "_harvest_incomplete_word_rows",
        lambda *_args, **_kwargs: [{"term": "project"}],
    )
    monkeypatch.setattr(
        growth_runtime, "_harvest_definition_seed_rows", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(growth_runtime, "_harvest_usage_seed_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(growth_runtime, "_harvest_emotion_seed_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        growth_runtime, "_harvest_derivation_seed_rows", lambda *_args, **_kwargs: []
    )

    class _FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, **kwargs):
            return [{"distance": 0.15, "metadata": {"term": "hola", "content_type": "word"}}]

    monkeypatch.setitem(growth_runtime.__dict__, "VectorStore", _FakeStore)

    class _FakeStorageType:
        DISK = "disk"

    monkeypatch.setitem(growth_runtime.__dict__, "StorageType", _FakeStorageType)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "word_forge.configs.config_essentials":
            mod = type("M", (), {"StorageType": _FakeStorageType})
            return mod
        if name == "word_forge.vectorizer.vector_store":
            mod = type("M", (), {"VectorStore": _FakeStore})
            return mod
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    rows = growth_runtime._harvest_vector_seed_rows(repo_root, db_path, limit=8)
    assert rows
    assert rows[0]["term"] == "hola"
    assert rows[0]["translation_thin"] is True
    assert rows[0]["translation_gap"] == 2
    assert rows[0]["language_count"] == 3
    assert rows[0]["translation_links"] == 1


def test_definition_function_words_are_kept_but_deprioritised() -> None:
    rows = growth_runtime._build_growth_priority_queue(
        forge_local_run={},
        lexicon_backfill_run={},
        multilingual_seed_rows=[],
        incomplete_seed_rows=[],
        definition_seed_rows=[
            {
                "term": "the",
                "mention_count": 100,
                "source_term_count": 80,
                "is_function_word": True,
            },
            {
                "term": "ontology",
                "mention_count": 5,
                "source_term_count": 4,
                "is_function_word": False,
            },
        ],
        limit=8,
    )
    assert {row["term"] for row in rows} == {"the", "ontology"}
    assert rows[0]["term"] == "ontology"


def test_harvest_definition_seed_rows_finds_missing_definition_terms(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word(
        "archive", "A durable storage registry for preserved records and provenance.", "noun", []
    )
    db.insert_or_update_word("storage", "The act of storing things.", "noun", [])
    db.close()

    rows = growth_runtime._harvest_definition_seed_rows(db_path, limit=12)
    terms = [row["term"] for row in rows]
    assert "the" in terms
    assert "registry" in terms
    assert "provenance" in terms
    assert "storage" not in terms


def test_valid_lexicon_term_allows_single_letter_words() -> None:
    assert growth_runtime._valid_lexicon_term("a") is True
    assert growth_runtime._valid_lexicon_term("i") is True
    assert growth_runtime._valid_lexicon_term("7") is False


def test_harvest_incomplete_word_rows_prioritises_missing_fields(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word("thin", "", "", [])
    db.insert_or_update_word("rich", "defined", "noun", [])
    rich_id = db.get_word_id("rich")
    db.execute_query(
        "INSERT INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, ?)",
        (rich_id, "wealthy", "synonym"),
    )
    db.close()

    rows = growth_runtime._harvest_incomplete_word_rows(db_path, limit=8)
    assert rows
    assert rows[0]["term"] == "thin"
    assert "definition" in rows[0]["missing_fields"]
    assert "derivation" in rows[0]["missing_fields"]


def test_lexicon_progress_snapshot_counts_core_fields(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = DBManager(db_path=db_path)
    db.create_tables()
    db.insert_or_update_word("hello", "greeting", "noun", [])
    word_id = db.get_word_id("hello")
    db.execute_query(
        "INSERT INTO relationships (word_id, related_term, relationship_type) VALUES (?, ?, ?)",
        (word_id, "hola", "translation"),
    )
    db.execute_query(
        "INSERT INTO phonetics (word_id, ipa, arpabet, stress_pattern, source, is_exception, last_updated) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
        (word_id, "həˈloʊ", None, None, "test", 0),
    )
    db.close()

    snapshot = growth_runtime._lexicon_progress_snapshot(db_path)
    assert snapshot["word_count"] == 1
    assert snapshot["defined_word_count"] == 1
    assert snapshot["pos_word_count"] == 1
    assert snapshot["phonetic_word_count"] == 1
    assert snapshot["translation_word_count"] == 1


def test_run_word_forge_growth_passes_synthesised_terms_to_thesaurus(
    monkeypatch, tmp_path: Path
) -> None:
    repo_root = tmp_path / "repo"
    db_path = repo_root / "word_forge" / "data" / "word_forge.sqlite"
    vector_db_path = repo_root / "data" / "word_forge_fasttext.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    vector_db_path.parent.mkdir(parents=True, exist_ok=True)
    DBManager(db_path=db_path).create_tables()

    monkeypatch.setattr(
        growth_runtime,
        "discover_growth_sources",
        lambda _repo_root: {
            "multilingual_sources": [],
            "fasttext_sources": [],
            "forge_local_sources": [],
            "multilingual_source_count": 0,
            "fasttext_source_count": 0,
            "forge_local_source_count": 0,
            "corpus_local_source_count": 0,
            "source_count": 0,
        },
    )
    monkeypatch.setattr(growth_runtime, "_semantic_vector_count", lambda _repo_root: 0)
    monkeypatch.setattr(
        growth_runtime,
        "run_forge_local_discovery",
        lambda **_kwargs: {
            "status": "completed",
            "inserted_count": 2,
            "inserted_terms": ["eidosian", "workflow"],
            "candidates": [{"term": "ontology"}],
        },
    )
    monkeypatch.setattr(
        growth_runtime,
        "run_polyglot_decomposition",
        lambda **_kwargs: {"status": {"decomposed_lexeme_delta": 0}},
    )
    monkeypatch.setattr(
        growth_runtime,
        "_harvest_multilingual_seed_rows",
        lambda *_args, **_kwargs: [
            {"term": "hola", "translation_links": 2, "language_count": 2},
            {"term": "bonjour", "translation_links": 2, "language_count": 2},
        ],
    )
    monkeypatch.setattr(
        growth_runtime,
        "_harvest_incomplete_word_rows",
        lambda *_args, **_kwargs: [
            {
                "term": "ontology",
                "missing_fields": ["phonetics"],
                "relationship_count": 1,
                "translation_count": 0,
            }
        ],
    )
    monkeypatch.setattr(growth_runtime, "_harvest_vector_seed_rows", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        growth_runtime,
        "run_lexicon_backfill",
        lambda **_kwargs: {
            "status": "completed",
            "inserted_word_count": 2,
            "updated_word_count": 0,
            "translation_relationship_delta": 4,
            "morpheme_link_delta": 0,
            "language_count": 2,
            "languages": ["en", "es"],
            "terms": ["hola", "bonjour"],
        },
    )
    monkeypatch.setattr(growth_runtime, "run_vector_index", lambda **_kwargs: True)
    monkeypatch.setattr(
        growth_runtime, "run_bridge_audit", lambda **_kwargs: {"status": {"fully_bridged": 0}}
    )
    monkeypatch.setattr(
        growth_runtime,
        "_write_growth_report",
        lambda repo_root, report: {
            "latest_json": str(repo_root / "reports" / "word_forge_growth" / "latest.json"),
            "latest_md": str(repo_root / "reports" / "word_forge_growth" / "latest.md"),
        },
    )

    captured: dict[str, object] = {}

    def _fake_thesaurus(**kwargs):
        captured.update(kwargs)
        return {
            "status": {"status": "completed"},
            "report": {
                "deltas": {"relationship_delta": 5, "phonetic_delta": 0, "omw_translation_delta": 7}
            },
        }

    monkeypatch.setattr(growth_runtime, "run_word_forge_thesaurus", _fake_thesaurus)

    result = growth_runtime.run_word_forge_growth(
        repo_root=repo_root,
        db_path=db_path,
        vector_db_path=vector_db_path,
        thesaurus_terms=["archive"],
        thesaurus_limit=12,
        lexicon_limit=12,
    )

    assert captured["seed_terms"] == [
        "archive",
        "eidosian",
        "workflow",
        "ontology",
        "hola",
        "bonjour",
    ]
    assert result["report"]["thesaurus_seed_term_count"] == 6
    assert result["report"]["multilingual_seed_term_count"] == 2
    assert result["report"]["incomplete_seed_term_count"] == 1
    assert result["report"]["omw_translation_delta"] == 7
    assert result["report"]["priority_queue"]
    assert result["report"]["priority_queue"][0]["term"] in {"hola", "bonjour"}
