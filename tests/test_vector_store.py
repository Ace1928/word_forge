import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class _StubCollection:
    def __init__(self, client):
        self.client = client

    def upsert(self, *args, **kwargs):  # pragma: no cover - stub
        return None

    def query(self, *args, **kwargs):  # pragma: no cover - stub
        return {"ids": [[]], "distances": [[]]}

    def delete(self, *args, **kwargs):  # pragma: no cover - stub
        return None


class _StubClient:
    def __init__(self):
        self.persist_called = False

    def get_or_create_collection(self, *_, **__):
        return _StubCollection(self)

    def persist(self):
        self.persist_called = True


chromadb_module = types.ModuleType("chromadb")
chromadb_module.Client = lambda *_, **__: _StubClient()
chromadb_module.PersistentClient = lambda *_, **__: _StubClient()
sys.modules["chromadb"] = chromadb_module


sentence_module = types.ModuleType("sentence_transformers")


class _DummyModel:
    def __init__(self, *_, **__):
        pass

    def get_sentence_embedding_dimension(self):
        return 5

    def encode(self, *_, **__):
        return np.zeros(5, dtype=np.float32)


sentence_module.SentenceTransformer = _DummyModel
sys.modules["sentence_transformers"] = sentence_module


faiss_module = types.ModuleType("faiss")


class _DummyIndexFlatIP:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self._vectors = np.zeros((0, dimension), dtype=np.float32)

    def add(self, vectors):
        array = np.asarray(vectors, dtype=np.float32)
        self._vectors = array.copy()

    def search(self, query, k):
        query_array = np.asarray(query, dtype=np.float32)
        if self._vectors.size == 0:
            distances = np.zeros((query_array.shape[0], k), dtype=np.float32)
            indices = -np.ones((query_array.shape[0], k), dtype=np.int64)
            return distances, indices
        similarities = self._vectors @ query_array[0]
        order = np.argsort(similarities)[::-1][:k]
        return similarities[order].reshape(1, -1), order.reshape(1, -1)


def _normalize_l2(arr):
    array = np.asarray(arr, dtype=np.float32)
    if array.ndim == 1:
        norm = np.linalg.norm(array) or 1.0
        array /= norm
    else:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        array /= norms
    return array


faiss_module.IndexFlatIP = _DummyIndexFlatIP
faiss_module.normalize_L2 = _normalize_l2
sys.modules["faiss"] = faiss_module

from word_forge.configs.config_essentials import StorageType
from word_forge.vectorizer.vector_store import (
    SQLITE_DB_FILENAME,
    DimensionMismatchError,
    InitializationError,
    SearchError,
    VectorStore,
)


class DummyCollection:
    def __init__(self, client):
        self.client = client

    def upsert(self, *a, **k):
        pass

    def query(self, *a, **k):
        return {"ids": [], "distances": []}

    def delete(self, *a, **k):
        pass


class DummyClient:
    def __init__(self):
        self.persist_called = False

    def get_or_create_collection(self, *a, **k):
        return DummyCollection(self)

    def persist(self):
        self.persist_called = True


def test_validate_vector_dimension_mismatch():
    vs = object.__new__(VectorStore)
    vs.dimension = 4
    with pytest.raises(DimensionMismatchError):
        vs._validate_vector_dimension(np.zeros(3, dtype=np.float32))


def test_search_requires_input():
    vs = object.__new__(VectorStore)
    with pytest.raises(SearchError):
        vs.search()


def test_persist_called_for_disk_storage():
    vs = object.__new__(VectorStore)
    vs.dimension = 5
    vs.client = DummyClient()
    vs.collection = DummyCollection(vs.client)
    vs.storage_type = StorageType.DISK
    vs._persist_if_needed()
    assert vs.client.persist_called


def test_demo_mode_requires_explicit_flag():
    with pytest.raises(InitializationError):
        VectorStore(dimension=5, storage_type=StorageType.MEMORY)

    store = VectorStore(dimension=5, storage_type=StorageType.MEMORY, demo_mode=True)
    assert store.demo_mode is True
    assert store.backend_name == "memory-demo"


def test_sqlite_faiss_fallback_used_when_chromadb_missing(tmp_path, monkeypatch):
    module = importlib.import_module("word_forge.vectorizer.vector_store")
    monkeypatch.setattr(module, "chromadb", None)

    store = module.VectorStore(
        dimension=5,
        storage_type=StorageType.DISK,
        index_path=tmp_path,
    )

    assert store.backend_name == "sqlite-faiss"
    store.upsert(1, np.zeros(5, dtype=np.float32))
    db_file = Path(tmp_path) / SQLITE_DB_FILENAME
    assert db_file.exists()

    results = store.search(query_vector=np.zeros(5, dtype=np.float32), k=1)
    assert len(results) == 1
    assert results[0]["id"] == 1
