from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from word_forge.forge import run_vector_index


class _FakeDB:
    def create_tables(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeWord:
    def __init__(self, word_id: int) -> None:
        self.id = word_id


def test_run_vector_index_skips_existing_sqlite_hnsw_words(monkeypatch, tmp_path: Path) -> None:
    from word_forge.vectorizer import vector_store as vector_store_module
    from word_forge.vectorizer import vector_worker as vector_worker_module

    index_path = tmp_path / "vector_indices"
    index_path.mkdir(parents=True, exist_ok=True)
    sqlite_path = index_path / vector_store_module.SQLITE_DB_FILENAME
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            "CREATE TABLE vectors (id TEXT PRIMARY KEY, embedding BLOB NOT NULL, metadata TEXT, document TEXT)"
        )
        conn.execute("CREATE TABLE collection_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO collection_meta(key, value) VALUES('backend', 'sqlite-hnsw')")
        conn.execute(
            "INSERT INTO vectors(id, embedding, metadata, document) VALUES (?, ?, ?, ?)",
            (
                "w_1",
                sqlite3.Binary(b"0" * 16),
                json.dumps({"original_id": 1, "term": "archive"}),
                "archive",
            ),
        )
        conn.commit()

    class FakeVectorStore:
        def __init__(self, db_manager=None, model_name=None):
            self.index_path = index_path
            self.backend_name = "sqlite-hnsw"

    processed_ids: list[int] = []

    class FakeVectorWorker:
        def __init__(
            self,
            db=None,
            vector_store=None,
            embedder=None,
            poll_interval=None,
            daemon=None,
            logger=None,
        ):
            self.stats = type(
                "Stats", (), {"successful": 0, "failed": 0, "clear": lambda self: None}
            )()
            self.last_processed = 0.0

        def _get_all_words(self):
            return [_FakeWord(1), _FakeWord(2)]

        def _process_words(self, chunk):
            processed_ids.extend(int(word.id) for word in chunk)
            self.stats.successful += len(chunk)

    monkeypatch.setattr(vector_store_module, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(vector_worker_module, "VectorWorker", FakeVectorWorker)

    assert run_vector_index(db_manager=_FakeDB(), timeout=5.0)
    assert processed_ids == [2]
