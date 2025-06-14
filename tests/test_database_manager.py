import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from word_forge.database.database_manager import DBManager, TermNotFoundError


def test_insert_and_get_word(tmp_path):
    db_path = tmp_path / "test.db"
    dbm = DBManager(db_path=db_path)
    dbm.insert_or_update_word("alpha", "first", "noun", ["example"])
    word_id = dbm.get_word_id("alpha")
    assert isinstance(word_id, int)


def test_get_word_id_not_found(tmp_path):
    db_path = tmp_path / "test.db"
    dbm = DBManager(db_path=db_path)
    dbm.create_tables()
    with pytest.raises(TermNotFoundError):
        dbm.get_word_id("missing")


def test_insert_word_empty_term(tmp_path):
    db_path = tmp_path / "test.db"
    dbm = DBManager(db_path=db_path)
    with pytest.raises(ValueError):
        dbm.insert_or_update_word("")
