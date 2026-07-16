"""CLI smoke tests without stubs or mocks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from word_forge.utils.nltk_utils import ensure_nltk_data

# Check if vector dependencies are available
_VECTOR_AVAILABLE = (
    importlib.util.find_spec("chromadb") is not None
    and importlib.util.find_spec("sentence_transformers") is not None
)

TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "sshleifer/tiny-gpt2"


def test_cli_version_command() -> None:
    from word_forge import forge

    assert forge.main(["--version"]) == 0


def test_cli_setup_nltk_command() -> None:
    from word_forge import forge

    ensure_nltk_data()
    assert forge.main(["setup-nltk"]) == 0


def test_cli_multilingual_setup_requires_license_acknowledgement() -> None:
    from word_forge import forge

    assert forge.main(["setup-nltk", "--multilingual"]) == 2


def test_cli_doctor_json_command(capsys: pytest.CaptureFixture[str]) -> None:
    from word_forge import forge

    ensure_nltk_data()
    assert forge.main(["doctor", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True


def test_cli_models_json_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Model readiness is discoverable without loading model weights."""
    from word_forge import forge

    assert forge.main(["models", "list", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["recommended"] in {
        "off",
        "portable",
        "gemma4-edge",
    }
    assert {profile["name"] for profile in report["profiles"]} == {
        "off",
        "portable",
        "gemma3-tiny",
        "gemma4-edge",
    }


def test_cli_sources_json_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Lexical source policy is discoverable without downloading data."""
    from word_forge import forge

    assert forge.main(["sources", "list", "--json"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1
    assert report["count"] >= 10
    assert {source["id"] for source in report["sources"]} >= {
        "cmudict",
        "panlex",
        "wikidata-lexemes",
    }


def test_cli_sources_unattended_filter(capsys: pytest.CaptureFixture[str]) -> None:
    """Automation excludes share-alike and per-dataset sources by default."""
    from word_forge import forge

    assert forge.main(["sources", "list", "--json", "--unattended-eligible"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["filters"]["unattended_only"] is True
    assert all(source["unattended_eligible"] for source in report["sources"])
    assert "dbnary" not in {source["id"] for source in report["sources"]}


def test_cli_kaikki_import_checkpoints_and_resumes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The public CLI performs real normalized writes and resumes safely."""

    from word_forge import forge
    from word_forge.database.database_manager import DBManager

    records = [
        {
            "id": "en-forge-noun",
            "word": "forge",
            "lang_code": "en",
            "pos": "noun",
            "senses": [{"glosses": ["A workshop containing a furnace."]}],
        },
        {
            "id": "en-forge-verb",
            "word": "forge",
            "lang_code": "en",
            "pos": "verb",
            "senses": [{"glosses": ["To shape metal by heating and hammering."]}],
        },
    ]
    artifact = tmp_path / "kaikki.jsonl"
    artifact.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    database_path = tmp_path / "kaikki.sqlite"
    base_arguments = [
        "data",
        "import-kaikki",
        str(artifact),
        "--source-version",
        "2026-07-06",
        "--source-url",
        "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz",
        "--expected-sha256",
        digest,
        "--accept-source-license",
        "--language",
        "en",
        "--batch-size",
        "1",
        "--db-path",
        str(database_path),
        "--json",
    ]

    assert forge.main(base_arguments + ["--max-entries", "1"]) == 0
    first = json.loads(capsys.readouterr().out)

    assert first["report"]["write_report"]["inserted"] == 1
    checkpoint = Path(first["checkpoint_path"])
    assert checkpoint == tmp_path / "kaikki.jsonl.word-forge.checkpoint.json"
    assert checkpoint.exists()

    assert forge.main(base_arguments) == 0
    resumed = json.loads(capsys.readouterr().out)

    assert resumed["report"]["first_line"] == 2
    assert resumed["report"]["write_report"]["inserted"] == 1
    database = DBManager(database_path)
    try:
        assert database.execute_scalar("SELECT COUNT(*) FROM lexical_entries") == 2
        assert database.execute_scalar("SELECT COUNT(*) FROM lexical_senses") == 2
    finally:
        database.close()


def test_cli_kaikki_import_requires_license_before_creating_database(
    tmp_path: Path,
) -> None:
    from word_forge import forge

    artifact = tmp_path / "kaikki.jsonl"
    artifact.write_text("{}\n", encoding="utf-8")
    database_path = tmp_path / "must-not-exist.sqlite"

    assert (
        forge.main(
            [
                "data",
                "import-kaikki",
                str(artifact),
                "--source-version",
                "2026-07-06",
                "--source-url",
                "https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz",
                "--db-path",
                str(database_path),
            ]
        )
        == 2
    )
    assert not database_path.exists()


def test_cli_start_core_without_vector_dependencies(tmp_path: Path) -> None:
    """The lightweight core pipeline must run without vector backends."""
    from word_forge import forge

    ensure_nltk_data()
    result = forge.main(
        [
            "start",
            "wordforgesmokenonword",
            "--minutes",
            "0.001",
            "--workers",
            "1",
            "--db-path",
            str(tmp_path / "cli_core_start.db"),
            "--no-vector",
        ]
    )

    assert result == 0


def test_cli_start_persists_requested_language(tmp_path: Path) -> None:
    """Language identity reaches SQLite even without optional OMW data."""
    from word_forge import forge
    from word_forge.database.database_manager import DBManager

    ensure_nltk_data()
    db_path = tmp_path / "cli_french_start.db"
    result = forge.main(
        [
            "start",
            "motforgenonexistent",
            "--language",
            "fr-FR",
            "--minutes",
            "0.001",
            "--workers",
            "1",
            "--db-path",
            str(db_path),
            "--no-vector",
        ]
    )

    assert result == 0
    database = DBManager(db_path=db_path)
    try:
        assert (
            database.get_word_entry("motforgenonexistent", "fr-FR")["language"]
            == "fr-FR"
        )
    finally:
        database.close()


@pytest.mark.skipif(
    not _VECTOR_AVAILABLE,
    reason="Vector dependencies (chromadb, sentence-transformers) not installed",
)
def test_cli_start_command(tmp_path: Path) -> None:
    from word_forge import forge

    ensure_nltk_data()

    result = forge.main(
        [
            "start",
            "happy",
            "--minutes",
            "0.01",
            "--workers",
            "1",
            "--db-path",
            str(tmp_path / "cli_start.db"),
            "--vector-model",
            TEST_MODEL,
            "--vector",
            "--llm-model",
            LLM_MODEL,
        ]
    )
    assert result == 0


def test_cli_graph_build_command(tmp_path: Path) -> None:
    from word_forge import forge

    assert (
        forge.main(
            [
                "graph",
                "build",
                "--timeout",
                "10",
                "--poll-interval",
                "0.5",
                "--db-path",
                str(tmp_path / "graph-build.sqlite"),
            ]
        )
        == 0
    )


def test_cli_graph_visualize_focused_standalone(tmp_path: Path) -> None:
    from word_forge import forge
    from word_forge.database.database_manager import DBManager

    database_path = tmp_path / "graph-view.sqlite"
    output_path = tmp_path / "focused.html"
    database = DBManager(db_path=database_path)
    database.insert_or_update_word("chat", language="fr")
    database.insert_or_update_word("bonjour", language="fr")
    database.insert_relationship(
        "chat",
        "bonjour",
        "related",
        base_language="fr",
        related_language="fr",
        source="cli-test",
    )
    database.close()

    result = forge.main(
        [
            "graph",
            "visualize",
            "--db-path",
            str(database_path),
            "--term",
            "chat",
            "--language",
            "fr",
            "--depth",
            "1",
            "--dimension",
            "lexical",
            "--max-nodes",
            "2",
            "--max-edges",
            "1",
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    assert output_path.is_file()
    rendered = output_path.read_text(encoding="utf-8")
    assert 'data-word-forge-viewer="1"' in rendered
    assert "Lexical connection graph" in rendered
    assert "cdnjs.cloudflare.com" not in rendered


@pytest.mark.skipif(
    not _VECTOR_AVAILABLE,
    reason="Vector dependencies (chromadb, sentence-transformers) not installed",
)
def test_cli_vector_index_command(tmp_path: Path) -> None:
    from word_forge import forge

    import os
    from word_forge.config import config
    db_file = str(tmp_path / "test_cli_vector.sqlite")
    
    old_db_path_env = os.environ.get("WORDFORGE_DB_PATH")
    os.environ["WORDFORGE_DB_PATH"] = db_file
    
    old_db_path_config = config.database.db_path
    config.database.db_path = db_file

    # Pre-populate the database with a word so the vectorizer has something to index
    from word_forge.database.database_manager import DBManager
    db = DBManager(db_path=db_file)
    db.create_tables()
    db.insert_or_update_word(
        term="hello",
        definition="a greeting",
        part_of_speech="noun",
        usage_examples=[],
        language="en-US",
        source="wiktionary",
        is_stub=False,
    )
    db.close()

    try:
        result = forge.main(
            [
                "vector",
                "index",
                "--embedder",
                TEST_MODEL,
                "--timeout",
                "120",
                "--poll-interval",
                "0.5",
            ]
        )
        assert result == 0
    finally:
        # Restore configuration and environment variable
        config.database.db_path = old_db_path_config
        if old_db_path_env is not None:
            os.environ["WORDFORGE_DB_PATH"] = old_db_path_env
        else:
            os.environ.pop("WORDFORGE_DB_PATH", None)
