"""Integration tests for word_forge.vectorizer.vector_store."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Skip all tests in this module if vector dependencies are unavailable
_VECTOR_AVAILABLE = (
    importlib.util.find_spec("chromadb") is not None
    and importlib.util.find_spec("sentence_transformers") is not None
)

pytestmark = pytest.mark.skipif(
    not _VECTOR_AVAILABLE,
    reason="Vector dependencies (chromadb, sentence-transformers) not installed",
)

from word_forge.configs.config_essentials import StorageType
from word_forge.vectorizer.vector_store import (
    DimensionMismatchError,
    InitializationError,
    SearchError,
    VectorStore,
)

TEST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _sample_entry() -> dict:
    return {
        "id": "1",
        "id_int": 1,
        "term": "happy",
        "definition": "feeling or showing pleasure",
        "part_of_speech": "adj",
        "usage_examples": ["I feel happy today."],
        "language": "en",
        "last_refreshed": 0.0,
        "relationships": [],
    }


class TestVectorStoreDemoMode:
    def test_demo_mode_requires_explicit_flag(self) -> None:
        with pytest.raises(InitializationError):
            VectorStore(
                dimension=384,
                storage_type=StorageType.MEMORY,
                model_name=TEST_MODEL,
            )

    def test_demo_mode_initialization(self) -> None:
        store = VectorStore(
            dimension=384,
            storage_type=StorageType.MEMORY,
            demo_mode=True,
            model_name=TEST_MODEL,
        )
        assert store.demo_mode is True
        assert store.backend_name == "memory-demo"


class TestVectorStoreBehavior:
    def test_search_requires_input(self) -> None:
        store = VectorStore(
            dimension=384,
            storage_type=StorageType.MEMORY,
            demo_mode=True,
            model_name=TEST_MODEL,
        )
        with pytest.raises(SearchError):
            store.search()

    def test_store_word_in_memory(self) -> None:
        store = VectorStore(
            dimension=384,
            storage_type=StorageType.MEMORY,
            demo_mode=True,
            model_name=TEST_MODEL,
        )
        stored = store.store_word(_sample_entry())
        assert stored > 0
        assert store.collection.count() > 0

    def test_initialization_rejects_dimension_override_for_other_model(self) -> None:
        with pytest.raises(DimensionMismatchError, match="does not match"):
            VectorStore(
                dimension=1024,
                storage_type=StorageType.MEMORY,
                demo_mode=True,
                model_name=TEST_MODEL,
            )

    def test_upsert_rejects_mismatched_dimension(self) -> None:
        store = VectorStore(
            dimension=384,
            storage_type=StorageType.MEMORY,
            demo_mode=True,
            model_name=TEST_MODEL,
        )

        with pytest.raises(DimensionMismatchError, match="expected 384"):
            store.upsert(
                "abc",
                np.ones(128, dtype=np.float32),
                metadata={"content_type": "word"},
            )

    def test_search_preserves_non_numeric_vector_ids(self) -> None:
        store = VectorStore(
            dimension=384,
            storage_type=StorageType.MEMORY,
            demo_mode=True,
            model_name=TEST_MODEL,
        )
        vector = np.ones(384, dtype=np.float32)
        store.upsert(
            "ja-definition",
            vector,
            metadata={"content_type": "definition", "language": "ja"},
            text="再帰的な言語の定義",
        )

        results = store.search(query_vector=vector, k=1)

        assert len(results) == 1
        assert results[0]["id"] == "ja-definition"
        assert results[0]["text"] == "再帰的な言語の定義"


class TestVectorStorePersistence:
    def test_disk_persistence_creates_files(self, tmp_path: Path) -> None:
        index_path = tmp_path / "vector_index"
        store = VectorStore(
            dimension=384,
            storage_type=StorageType.DISK,
            model_name=TEST_MODEL,
            index_path=index_path,
            collection_name="test_collection",
        )
        store.store_word(_sample_entry())

        persisted = list(index_path.rglob("*.sqlite*"))
        assert persisted, "Expected vector store persistence files to be created"
