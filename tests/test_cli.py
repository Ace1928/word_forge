"""CLI smoke tests without stubs or mocks."""

from __future__ import annotations

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


def test_cli_graph_build_command() -> None:
    from word_forge import forge

    assert (
        forge.main(["graph", "build", "--timeout", "10", "--poll-interval", "0.5"]) == 0
    )


@pytest.mark.skipif(
    not _VECTOR_AVAILABLE,
    reason="Vector dependencies (chromadb, sentence-transformers) not installed",
)
def test_cli_vector_index_command(tmp_path: Path) -> None:
    from word_forge import forge

    result = forge.main(
        [
            "vector",
            "index",
            "--embedder",
            TEST_MODEL,
            "--timeout",
            "20",
            "--poll-interval",
            "0.5",
        ]
    )
    assert result == 0
