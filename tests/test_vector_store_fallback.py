from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

import word_forge.vectorizer.vector_store as vector_store
from word_forge.configs.config_essentials import StorageType
from word_forge.vectorizer.vector_store import ContentProcessingError, ModelLoadError


def test_vector_store_prefers_sqlite_hnsw_when_available(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(vector_store, "SentenceTransformer", None)
    monkeypatch.setattr(vector_store, "chromadb", None)
    monkeypatch.setattr(vector_store, "faiss", None)

    index_path = tmp_path / "vector_index"
    store = vector_store.VectorStore(
        dimension=8,
        storage_type=StorageType.DISK,
        model_name="dummy-local-fallback",
        index_path=index_path,
        collection_name="fallback_test",
    )

    expected_backend = (
        "sqlite-hnsw" if getattr(vector_store, "hnswlib", None) is not None else "sqlite-linear"
    )
    assert store.backend_name == expected_backend

    store.upsert(
        vec_id="w_1",
        embedding=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        metadata={"content_type": "word", "term": "archive", "language": "en", "original_id": 1},
        text="archive",
    )
    store.upsert(
        vec_id="w_2",
        embedding=np.asarray([0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        metadata={"content_type": "word", "term": "atlas", "language": "en", "original_id": 2},
        text="atlas",
    )

    results = store.search(
        query_vector=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        k=2,
        filter_metadata={"content_type": "word"},
    )

    assert len(results) == 2
    assert results[0]["metadata"]["term"] == "archive"
    assert results[1]["metadata"]["term"] == "atlas"

    sqlite_path = index_path / vector_store.SQLITE_DB_FILENAME
    with sqlite3.connect(sqlite_path) as conn:
        backend = conn.execute("SELECT value FROM collection_meta WHERE key = 'backend'").fetchone()
        count = conn.execute("SELECT COUNT(*) FROM vectors").fetchone()

    assert backend == (expected_backend,)
    assert count == (2,)


def test_storage_only_mode_never_fabricates_embeddings(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(vector_store, "SentenceTransformer", None)

    with pytest.raises(ModelLoadError, match="sentence-transformers"):
        vector_store.VectorStore(
            storage_type=StorageType.DISK,
            model_name="unavailable-model",
            index_path=tmp_path / "missing_dimension",
        )

    store = vector_store.VectorStore(
        dimension=8,
        storage_type=StorageType.DISK,
        model_name="unavailable-model",
        index_path=tmp_path / "storage_only",
    )
    with pytest.raises(ContentProcessingError, match="storage only"):
        store.embed_text("never invent semantic data")


def test_vector_store_uses_sqlite_linear_when_all_ann_backends_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(vector_store, "SentenceTransformer", None)
    monkeypatch.setattr(vector_store, "chromadb", None)
    monkeypatch.setattr(vector_store, "faiss", None)
    monkeypatch.setattr(vector_store, "hnswlib", None)

    index_path = tmp_path / "vector_index_linear"
    store = vector_store.VectorStore(
        dimension=8,
        storage_type=StorageType.DISK,
        model_name="dummy-local-linear",
        index_path=index_path,
        collection_name="fallback_test_linear",
    )

    assert store.backend_name == "sqlite-linear"
