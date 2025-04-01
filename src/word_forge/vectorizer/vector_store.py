from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, Union, cast

import chromadb
import numpy as np

# Type definitions for clarity and constraint
Vector = np.ndarray  # Represents embedding vectors
VectorID = int  # Unique identifier for vectors
SearchResult = List[Tuple[VectorID, float]]  # (id, distance) pairs
Metadata = Dict[str, any]  # ChromaDB metadata type


class ChromaCollection(Protocol):
    """Protocol defining required ChromaDB collection interface."""

    def count(self) -> int:
        """Return number of items in collection."""
        ...

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Metadata]] = None,
    ) -> None:
        """Add or update items in collection."""
        ...

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 10,
    ) -> Dict[str, List[any]]:
        """Query collection with embeddings."""
        ...


class ChromaClient(Protocol):
    """Protocol defining required ChromaDB client interface."""

    def get_or_create_collection(
        self, name: str, metadata: Optional[Metadata] = None
    ) -> ChromaCollection:
        """Get or create a collection."""
        ...

    # Note: persist() may not exist on all client types


class StorageType(Enum):
    """Storage strategy for vector embeddings."""

    MEMORY = "memory"
    DISK = "disk"


class VectorStoreError(Exception):
    """Base exception for vector store operations."""

    pass


class InitializationError(VectorStoreError):
    """Raised when vector store initialization fails."""

    pass


class UpsertError(VectorStoreError):
    """Raised when adding or updating vectors fails."""

    pass


class SearchError(VectorStoreError):
    """Raised when vector similarity search fails."""

    pass


class DimensionMismatchError(ValueError):
    """Raised when vector dimensions don't match expected dimensions."""

    pass


class VectorStore:
    """
    Vector store for embeddings using ChromaDB with flexible storage options.

    Provides a high-performance vector database for similarity search with
    support for both in-memory and persistent storage options.
    """

    def __init__(
        self,
        dimension: int,
        index_path: Union[str, Path] = "data/vector.index",
        storage_type: StorageType = StorageType.DISK,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the vector store with specified configuration.

        Args:
            dimension: Dimensionality of vectors to be stored
            index_path: Path for persistent storage (used with DISK storage)
            storage_type: Whether to use in-memory or disk-based storage
            collection_name: Name for the collection (defaults to index filename)

        Raises:
            InitializationError: If ChromaDB initialization fails
        """
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.storage_type = storage_type
        self._has_persist_method = False  # Track if client has persist method

        # Determine collection name from path if not provided
        if collection_name is None:
            collection_name = self.index_path.stem or "default_collection"

        try:
            self.client = self._create_client()
            # Check if client has persist method
            self._has_persist_method = hasattr(self.client, "persist") and callable(
                getattr(self.client, "persist")
            )
            self.collection = self._initialize_collection(collection_name)
        except Exception as e:
            raise InitializationError(
                f"ChromaDB initialization failed: {str(e)}"
            ) from e

    def _create_client(self) -> ChromaClient:
        """
        Create ChromaDB client based on storage type configuration.

        Returns:
            Configured ChromaDB client for the selected storage type
        """
        if self.storage_type == StorageType.MEMORY:
            return cast(ChromaClient, chromadb.EphemeralClient())

        # DISK storage
        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        return cast(
            ChromaClient, chromadb.PersistentClient(path=str(self.index_path.parent))
        )

    def _initialize_collection(self, collection_name: str) -> ChromaCollection:
        """
        Initialize ChromaDB collection with metadata.

        Args:
            collection_name: Name for the collection

        Returns:
            Initialized ChromaDB collection
        """
        return self.client.get_or_create_collection(
            name=collection_name, metadata={"dimension": self.dimension}
        )

    def _validate_vector_dimension(
        self, vector: Vector, context: str = "Vector"
    ) -> None:
        """
        Validate that vector has the expected dimension.

        Args:
            vector: Vector to validate
            context: Context string for error message

        Raises:
            DimensionMismatchError: If dimensions don't match
        """
        if vector.shape != (self.dimension,):
            actual_dim = vector.shape[0] if len(vector.shape) > 0 else 0
            raise DimensionMismatchError(
                f"{context} dimension mismatch. Expected {self.dimension}, got {actual_dim}"
            )

    def upsert(self, vec_id: VectorID, embedding: Vector) -> None:
        """
        Add or update a vector in the store.

        Args:
            vec_id: Unique identifier for the vector
            embedding: Vector embedding to store

        Raises:
            DimensionMismatchError: If embedding dimension doesn't match expected dimension
            UpsertError: If ChromaDB operation fails
        """
        self._validate_vector_dimension(embedding, "Embedding")

        try:
            # ChromaDB expects string IDs and list embeddings
            str_id = str(vec_id)
            embedding_list = embedding.tolist()

            self.collection.upsert(
                ids=[str_id],
                embeddings=[embedding_list],
                metadatas=[{"original_id": vec_id}],
            )

            # Persist changes for disk-based storage
            self._persist_if_needed()
        except Exception as e:
            raise UpsertError(f"Vector upsert failed: {str(e)}") from e

    def search(self, query_vector: Vector, k: int = 5) -> SearchResult:
        """
        Find k most similar vectors to the query.

        Args:
            query_vector: Query embedding to search against
            k: Number of results to return (limited by collection size)

        Returns:
            List of tuples (id, distance) sorted by similarity (lowest distance first)

        Raises:
            DimensionMismatchError: If query vector dimension doesn't match expected dimension
            SearchError: If search operation fails
        """
        self._validate_vector_dimension(query_vector, "Query vector")

        # For backward compatibility with FAISS interface
        if self.collection.count() == 0:
            return []

        try:
            # Perform search with limit based on available items
            effective_k = min(k, self.collection.count())
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=effective_k,
            )

            # No results case
            if not results["ids"][0]:
                return []

            # Process results
            ids = [int(id_str) for id_str in results["ids"][0]]
            distances = self._convert_similarities_to_distances(results["distances"][0])

            return list(zip(ids, distances))
        except Exception as e:
            raise SearchError(f"Vector search failed: {str(e)}") from e

    def _convert_similarities_to_distances(
        self, similarities: List[float]
    ) -> List[float]:
        """
        Convert ChromaDB similarities to distance metrics.

        ChromaDB returns similarity scores where 1.0 is perfect match.
        For compatibility with FAISS, convert to distances where 0.0 is perfect match.

        Args:
            similarities: List of similarity scores from ChromaDB

        Returns:
            List of distance scores (1.0 - similarity)
        """
        return [1.0 - similarity for similarity in similarities]

    def _persist_if_needed(self) -> None:
        """
        Persist to disk if using disk storage and client supports it.

        This safely handles clients that may not have a persist method.
        """
        if self.storage_type == StorageType.DISK and self._has_persist_method:
            try:
                # Only call persist if the method exists
                getattr(self.client, "persist")()
            except Exception as e:
                # Log but don't crash on persistence failure
                import logging

                logging.warning(f"Failed to persist vector store: {e}")

    def __del__(self) -> None:
        """Ensure persistent storage is properly finalized."""
        try:
            if (
                hasattr(self, "client")
                and self.storage_type == StorageType.DISK
                and self._has_persist_method
            ):
                getattr(self.client, "persist")()
        except Exception:
            # Silently continue with destruction if cleanup fails
            # This prevents errors during garbage collection
            pass


if __name__ == "__main__":
    # Example usage demonstrating vector store functionality

    # Create a 768-dimensional vector store with in-memory storage
    vector_store = VectorStore(dimension=768, storage_type=StorageType.MEMORY)

    # Create some example embeddings
    embeddings = [
        np.random.rand(768).astype(np.float32),  # Random vector 1
        np.random.rand(768).astype(np.float32),  # Random vector 2
        np.random.rand(768).astype(np.float32),  # Random vector 3
    ]

    # Store the embeddings
    for i, embedding in enumerate(embeddings):
        vector_store.upsert(vec_id=i, embedding=embedding)

    # Search for the most similar vector to the first embedding
    results = vector_store.search(query_vector=embeddings[0], k=2)

    # Display results
    print("Search results for vector 0:")
    for vec_id, distance in results:
        print(f"  Vector ID: {vec_id}, Distance: {distance:.6f}")
