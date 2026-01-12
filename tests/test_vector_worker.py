"""Tests for word_forge.vectorizer.vector_worker module.

This module tests the VectorWorker class which handles vectorization of words.
Note: Some tests require simplified implementations due to ML model dependencies.
"""

import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from word_forge.database.database_manager import DBManager
from word_forge.vectorizer.vector_worker import EmbeddingError


class SimpleEmbedder:
    """A simple embedder for testing that doesn't require ML models."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def embed(self, text: str) -> np.ndarray:
        """Create a simple hash-based embedding for testing."""
        # Create a deterministic embedding based on text hash
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.dimension).astype(np.float32)


class SimpleVectorStore:
    """A simple in-memory vector store for testing."""

    def __init__(self):
        self.vectors = {}
        self.ids = []

    def upsert(self, id_: int, vector: np.ndarray) -> None:
        """Store a vector with its ID."""
        self.vectors[id_] = vector
        self.ids.append(id_)

    def get(self, id_: int) -> np.ndarray:
        """Retrieve a vector by ID."""
        return self.vectors.get(id_)


def test_word_processing_tracks_modified_words(tmp_path):
    """Test that the worker correctly tracks which words have been modified."""
    from word_forge.vectorizer.vector_worker import VectorWorker

    db = DBManager(db_path=tmp_path / "test.db")
    db.insert_or_update_word("alpha", "first")
    db.insert_or_update_word("beta", "second")

    store = SimpleVectorStore()
    embedder = SimpleEmbedder()
    worker = VectorWorker(db, store, embedder)

    # Process all words
    words = worker._get_all_words()
    worker._process_words(words)
    worker.last_processed = time.time()

    first_ids = list(store.ids)
    assert len(first_ids) == 2

    # Add new word and update existing
    time.sleep(0.01)
    db.insert_or_update_word("gamma", "third")
    db.insert_or_update_word("alpha", "updated")

    # Get words modified since last processing
    words2 = worker._get_all_words()
    terms2 = {w.term for w in words2}
    assert terms2 == {"gamma", "alpha"}

    # Process the modified words
    worker._process_words(words2)

    # Verify new words were added
    assert len(store.ids) > len(first_ids)
    assert db.get_word_id("gamma") in store.ids
    assert db.get_word_id("alpha") in store.ids


def test_embedder_produces_consistent_vectors():
    """Test that the simple embedder produces consistent vectors."""
    embedder = SimpleEmbedder(dimension=64)

    vec1 = embedder.embed("hello")
    vec2 = embedder.embed("hello")

    # Same text should produce same vector
    np.testing.assert_array_equal(vec1, vec2)

    # Different text should produce different vector
    vec3 = embedder.embed("world")
    assert not np.array_equal(vec1, vec3)


def test_vector_store_upsert_and_retrieve():
    """Test that vectors can be stored and retrieved."""
    store = SimpleVectorStore()
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    store.upsert(1, vector)
    retrieved = store.get(1)

    np.testing.assert_array_equal(vector, retrieved)


def test_vector_store_tracks_ids():
    """Test that the store tracks all inserted IDs."""
    store = SimpleVectorStore()

    store.upsert(1, np.zeros(3))
    store.upsert(2, np.zeros(3))
    store.upsert(3, np.zeros(3))

    assert store.ids == [1, 2, 3]


class TestTransformerEmbedderErrorHandling:
    """Test error handling for TransformerEmbedder initialization."""

    def test_import_error_has_helpful_message(self):
        """Test that ImportError gives helpful installation instructions."""
        from word_forge.vectorizer.vector_worker import TransformerEmbedder

        # Mock sentence_transformers import to raise ImportError
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with patch(
                "word_forge.vectorizer.vector_worker.TransformerEmbedder.__init__"
            ) as mock_init:
                # Simulate the ImportError path
                mock_init.side_effect = EmbeddingError(
                    "Missing required package: sentence-transformers.\n"
                    "To fix this, install the vector dependencies:\n"
                    '    pip install "word_forge[vector]"'
                )
                with pytest.raises(EmbeddingError) as exc_info:
                    TransformerEmbedder("test-model")

                error_msg = str(exc_info.value)
                assert "sentence-transformers" in error_msg
                assert "pip install" in error_msg

    def test_model_not_found_error_has_helpful_suggestions(self):
        """Test that model not found errors provide helpful suggestions."""
        from word_forge.vectorizer.vector_worker import TransformerEmbedder

        with patch(
            "word_forge.vectorizer.vector_worker.TransformerEmbedder.__init__"
        ) as mock_init:
            # Simulate a model not found error
            mock_init.side_effect = EmbeddingError(
                "Model not found: 'invalid-model'.\n"
                "This model name is not recognized by HuggingFace.\n"
                "Popular embedding models:\n"
                "  - sentence-transformers/all-MiniLM-L6-v2"
            )
            with pytest.raises(EmbeddingError) as exc_info:
                TransformerEmbedder("invalid-model")

            error_msg = str(exc_info.value)
            assert (
                "not found" in error_msg.lower()
                or "not recognized" in error_msg.lower()
            )
            assert "sentence-transformers/all-MiniLM-L6-v2" in error_msg

    def test_embedding_error_is_propagated_from_vector_worker(self):
        """Test that EmbeddingError is properly propagated from VectorWorker."""
        from word_forge.vectorizer.vector_worker import VectorWorker

        db = MagicMock()
        store = MagicMock()

        # Mock TransformerEmbedder to raise an EmbeddingError
        with patch(
            "word_forge.vectorizer.vector_worker.TransformerEmbedder"
        ) as mock_embedder:
            mock_embedder.side_effect = EmbeddingError("Test embedding error")

            with pytest.raises(EmbeddingError) as exc_info:
                VectorWorker(db, store, "some-model")

            assert "Test embedding error" in str(exc_info.value)

    def test_vector_worker_accepts_custom_embedder(self):
        """Test that VectorWorker accepts a custom embedder object."""
        from word_forge.vectorizer.vector_worker import VectorWorker

        db = MagicMock()
        store = MagicMock()
        embedder = SimpleEmbedder(dimension=64)

        # This should not raise any error
        worker = VectorWorker(db, store, embedder)
        assert worker.embedder is embedder
        assert worker.embedder.dimension == 64
