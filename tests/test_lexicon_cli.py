"""End-to-end CLI tests for the approachable local lexicon workflow."""

from __future__ import annotations

import json
from pathlib import Path

from word_forge import forge


def test_cli_local_lexicon_workflow(tmp_path: Path, capsys) -> None:
    database = tmp_path / "words.sqlite"

    assert forge.main(["init", "--db-path", str(database)]) == 0
    assert (
        forge.main(
            [
                "add",
                "recursion",
                "--definition",
                "A process that refers to itself.",
                "--db-path",
                str(database),
            ]
        )
        == 0
    )
    assert forge.main(["show", "recursion", "--db-path", str(database), "--json"]) == 0

    output = capsys.readouterr().out
    payload = json.loads(output[output.index("{") :])
    assert payload["term"] == "recursion"
    assert payload["language"] == "en"


def test_cli_find_translate_and_stats_json(tmp_path: Path, capsys) -> None:
    database = tmp_path / "words.sqlite"
    common = ["--db-path", str(database)]

    assert forge.main(["add", "house", "--definition", "A home.", *common]) == 0
    assert (
        forge.main(
            [
                "translate",
                "house",
                "Haus",
                "--from",
                "en",
                "--to",
                "de",
                *common,
            ]
        )
        == 0
    )
    assert forge.main(["find", "Haus", "--language", "de", *common, "--json"]) == 0
    find_output = capsys.readouterr().out
    assert '"term": "Haus"' in find_output

    assert forge.main(["stats", *common, "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["words"] == 1
    assert stats["lexemes"] == 2
    assert stats["translations"] == 2


def test_cli_missing_entry_has_stable_exit_code(tmp_path: Path, capsys) -> None:
    result = forge.main(["show", "missing", "--db-path", str(tmp_path / "words.sqlite")])
    assert result == 3
    assert "ERROR:" in capsys.readouterr().err


def test_cli_senses_and_revision_conflicts(tmp_path: Path, capsys) -> None:
    database = tmp_path / "words.sqlite"
    common = ["--db-path", str(database)]
    assert forge.main(["add", "bank", "--if-revision", "0", *common, "--json"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["revision"] == 1
    assert (
        forge.main(
            [
                "sense",
                "bank",
                "A financial institution.",
                "--domain",
                "finance",
                "--if-revision",
                "0",
                *common,
                "--json",
            ]
        )
        == 0
    )
    sense = json.loads(capsys.readouterr().out)
    assert sense["domain"] == "finance"
    assert (
        forge.main(
            [
                "add",
                "bank",
                "--definition",
                "stale",
                "--if-revision",
                "0",
                *common,
            ]
        )
        == 4
    )
    assert "Revision conflict" in capsys.readouterr().err


def test_cli_integrity_backup_and_guarded_restore(tmp_path: Path, capsys) -> None:
    database = tmp_path / "words.sqlite"
    backup = tmp_path / "snapshot.sqlite"
    common = ["--db-path", str(database)]
    assert forge.main(["add", "durable", *common]) == 0
    capsys.readouterr()
    assert forge.main(["integrity", *common, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert forge.main(["backup", str(backup), *common, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["backup"] == str(backup.resolve())
    assert forge.main(["restore", str(backup), *common]) == 2
    assert "--force" in capsys.readouterr().err
    assert forge.main(["restore", str(backup), *common, "--force"]) == 0


def test_doctor_json_reports_required_checks(capsys) -> None:
    assert forge.main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert {item["name"] for item in payload["checks"]} >= {
        "data-directory",
        "sqlite-fts5",
    }


def test_gui_command_forwards_validated_local_options(monkeypatch, tmp_path: Path) -> None:
    from word_forge.web import server

    captured = {}

    def fake_serve(**options):
        captured.update(options)
        return 0

    monkeypatch.setattr(server, "serve", fake_serve)
    database = tmp_path / "gui.sqlite"
    assert (
        forge.main(
            [
                "gui",
                "--db-path",
                str(database),
                "--port",
                "9876",
                "--no-open",
            ]
        )
        == 0
    )
    assert captured == {
        "database_path": database,
        "host": "127.0.0.1",
        "port": 9876,
        "open_browser": False,
    }
