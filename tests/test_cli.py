"""CLI smoke tests without stubs or mocks."""

from __future__ import annotations

from pathlib import Path

from word_forge.utils.nltk_utils import ensure_nltk_data

TEST_MODEL = "unit-test-no-vector-load"


def test_cli_version_command() -> None:
    from word_forge import forge

    assert forge.main(["--version"]) == 0


def test_cli_setup_nltk_command() -> None:
    from word_forge import forge

    ensure_nltk_data()
    assert forge.main(["setup-nltk"]) == 0


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
            "--no-vector",
            "--no-llm",
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
                str(tmp_path / "cli_graph.db"),
            ]
        )
        == 0
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
            "--db-path",
            str(tmp_path / "cli_vector.db"),
        ]
    )
    assert result == 0
