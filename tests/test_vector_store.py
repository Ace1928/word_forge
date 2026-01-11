"""Tests for word_forge.vectorizer.vector_store module.

This module tests the VectorStore class and its various backends.
Note: Some tests use mocks for external dependencies (chromadb, sentence-transformers)
that require network access to load models.
"""

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# Create mock implementations for external dependencies that require network
class MockCollection:
    """Mock ChromaDB collection for testing."""

    def __init__(self, client):
        self.client = client
        self.data = {}

    def upsert(self, ids=None, embeddings=None, **kwargs):
        if ids and embeddings:
            for id_, emb in zip(ids, embeddings):
                self.data[id_] = emb

    def query(self, query_embeddings=None, n_results=1, **kwargs):
        # Return empty results for testing
        return {"ids": [[]], "distances": [[]]}

    def delete(self, ids=None, **kwargs):
        if ids:
            for id_ in ids:
                self.data.pop(id_, None)


class MockClient:
    """Mock ChromaDB client for testing."""

    def __init__(self):
        self.persist_called = False
        self.collection = None

    def get_or_create_collection(self, name=None, **kwargs):
        self.collection = MockCollection(self)
        return self.collection

    def persist(self):
        self.persist_called = True


class MockSentenceTransformer:
    """Mock SentenceTransformer for testing without network access."""

    def __init__(self, model_name=None, **kwargs):
        self.model_name = model_name

    def get_sentence_embedding_dimension(self):
        return 5

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.zeros(5, dtype=np.float32)
        return np.zeros((len(texts), 5), dtype=np.float32)


class MockIndexFlatIP:
    """Mock FAISS index for testing."""

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
    """Mock L2 normalization function."""
    array = np.asarray(arr, dtype=np.float32)
    if array.ndim == 1:
        norm = np.linalg.norm(array) or 1.0
        array /= norm
    else:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        array /= norms
    return array


# Set up module-level mocks for imports
chromadb_module = types.ModuleType("chromadb")
chromadb_module.Client = lambda *_, **__: MockClient()
chromadb_module.PersistentClient = lambda *_, **__: MockClient()
sys.modules["chromadb"] = chromadb_module

sentence_module = types.ModuleType("sentence_transformers")
sentence_module.SentenceTransformer = MockSentenceTransformer
sys.modules["sentence_transformers"] = sentence_module

faiss_module = types.ModuleType("faiss")
faiss_module.IndexFlatIP = MockIndexFlatIP
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


class TestVectorStoreValidation:
    """Tests for VectorStore validation methods."""

    def test_validate_vector_dimension_mismatch(self):
        """Test that dimension mismatch raises appropriate error."""
        vs = object.__new__(VectorStore)
        vs.dimension = 4
        with pytest.raises(DimensionMismatchError):
            vs._validate_vector_dimension(np.zeros(3, dtype=np.float32))

    def test_validate_vector_dimension_match(self):
        """Test that matching dimensions pass validation."""
        vs = object.__new__(VectorStore)
        vs.dimension = 4
        # Should not raise
        vs._validate_vector_dimension(np.zeros(4, dtype=np.float32))


class TestVectorStoreSearch:
    """Tests for VectorStore search functionality."""

    def test_search_requires_input(self):
        """Test that search requires at least one input."""
        vs = object.__new__(VectorStore)
        with pytest.raises(SearchError):
            vs.search()


class TestVectorStorePersistence:
    """Tests for VectorStore persistence functionality."""

    def test_persist_called_for_disk_storage(self):
        """Test that persist is called for disk storage."""
        vs = object.__new__(VectorStore)
        vs.dimension = 5
        vs.client = MockClient()
        vs.collection = MockCollection(vs.client)
        vs.storage_type = StorageType.DISK
        vs._persist_if_needed()
        assert vs.client.persist_called

    def test_persist_not_called_for_memory_storage(self):
        """Test that persist is not called for memory storage."""
        vs = object.__new__(VectorStore)
        vs.dimension = 5
        vs.client = MockClient()
        vs.collection = MockCollection(vs.client)
        vs.storage_type = StorageType.MEMORY
        vs._persist_if_needed()
        assert not vs.client.persist_called


class TestVectorStoreDemoMode:
    """Tests for VectorStore demo mode."""

    def test_demo_mode_requires_explicit_flag(self):
        """Test that demo mode requires explicit flag."""
        with pytest.raises(InitializationError):
            VectorStore(dimension=5, storage_type=StorageType.MEMORY)

    def test_demo_mode_initialization(self):
        """Test demo mode initializes correctly."""
        store = VectorStore(
            dimension=5, storage_type=StorageType.MEMORY, demo_mode=True
        )
        assert store.demo_mode is True
        assert store.backend_name == "memory-demo"


class TestVectorStoreFallback:
    """Tests for VectorStore SQLite-FAISS fallback."""

    def test_sqlite_faiss_fallback_used_when_chromadb_missing(
        self, tmp_path, monkeypatch
    ):
        """Test that SQLite-FAISS backend is used when chromadb is unavailable."""
        module = importlib.import_module("word_forge.vectorizer.vector_store")
        monkeypatch.setattr(module, "chromadb", None)
        # Ensure faiss mock is available in the module for the fallback to work
        monkeypatch.setattr(module, "faiss", faiss_module)

        store = module.VectorStore(
            dimension=5,
            storage_type=StorageType.DISK,
            index_path=tmp_path,
        )

        assert store.backend_name == "sqlite-faiss"

        # Test basic operations
        store.upsert(1, np.zeros(5, dtype=np.float32))
        db_file = Path(tmp_path) / SQLITE_DB_FILENAME
        assert db_file.exists()

        results = store.search(query_vector=np.zeros(5, dtype=np.float32), k=1)
        assert len(results) == 1
        assert results[0]["id"] == 1
