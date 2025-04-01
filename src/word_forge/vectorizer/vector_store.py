from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from word_forge.config import config
from word_forge.database.db_manager import DatabaseError, DBManager, WordEntryDict
from word_forge.emotion.emotion_manager import EmotionManager

# Type definitions for clarity and constraint
Vector = np.ndarray  # Represents embedding vectors
VectorID = int  # Unique identifier for vectors
SearchResult = List[Tuple[VectorID, float]]  # (id, distance) pairs
ContentType = Literal["word", "definition", "example", "message", "conversation"]


class VectorMetadata(TypedDict, total=False):
    """Metadata associated with stored vectors."""

    original_id: int
    content_type: ContentType
    term: Optional[str]  # Term associated with this vector if it represents a word
    definition: Optional[str]  # Definition if this represents a definition
    speaker: Optional[str]  # Speaker if this is a conversation message
    emotion_valence: Optional[float]  # Emotional valence if available
    emotion_arousal: Optional[float]  # Emotional arousal if available
    emotion_label: Optional[str]  # Emotion classification if available
    conversation_id: Optional[int]  # Conversation ID if this is a message
    timestamp: Optional[float]  # When this content was created/processed
    language: Optional[str]  # Language of the content


class SearchResultDict(TypedDict):
    """Type definition for search result items."""

    id: int
    distance: float
    metadata: Optional[VectorMetadata]
    text: Optional[str]


class InstructionTemplate(TypedDict):
    """Type definition for instruction template."""

    task: str
    query_prefix: str
    document_prefix: Optional[str]


class ChromaCollection(Protocol):
    """Protocol defining required ChromaDB collection interface."""

    def count(self) -> int: ...
    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None: ...
    def query(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        query_texts: Optional[List[str]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, any]] = None,
    ) -> Dict[str, List[any]]: ...


class ChromaClient(Protocol):
    """Protocol defining required ChromaDB client interface."""

    def get_or_create_collection(
        self, name: str, metadata: Optional[Dict[str, any]] = None
    ) -> ChromaCollection: ...


# Using StorageType from centralized config
StorageType = config.vectorizer.storage_type.__class__


class VectorStoreError(DatabaseError):
    """Base exception for vector store operations."""

    pass


class InitializationError(VectorStoreError):
    """Raised when vector store initialization fails."""

    pass


class ModelLoadError(VectorStoreError):
    """Raised when embedding model loading fails."""

    pass


class UpsertError(VectorStoreError):
    """Raised when adding or updating vectors fails."""

    pass


class SearchError(VectorStoreError):
    """Raised when vector similarity search fails."""

    pass


class DimensionMismatchError(VectorStoreError):
    """Raised when vector dimensions don't match expected dimensions."""

    pass


class ContentProcessingError(VectorStoreError):
    """Raised when processing content for embedding fails."""

    pass


# SQL query constants from centralized config
SQL_GET_TERM_BY_ID = config.vectorizer.sql_templates["get_term_by_id"]
SQL_GET_MESSAGE_TEXT = config.vectorizer.sql_templates["get_message_text"]


class VectorStore:
    """
    Universal vector store for all Word Forge linguistic data with multilingual support.

    Provides storage and retrieval for embeddings of diverse linguistic content:
    - Words and their definitions in multiple languages
    - Usage examples
    - Conversation messages
    - Emotional valence and arousal

    Uses the advanced Multilingual-E5-large-instruct model to create contextually rich
    embeddings with instruction-based formatting for optimal retrieval performance.
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        model_name: Optional[str] = None,
        index_path: Optional[Union[str, Path]] = None,
        storage_type: Optional[StorageType] = None,
        collection_name: Optional[str] = None,
        db_manager: Optional[DBManager] = None,
        emotion_manager: Optional[EmotionManager] = None,
    ):
        """
        Initialize the vector store with specified configuration.

        Args:
            dimension: Optional override for vector dimensions
            model_name: Name of the SentenceTransformer model to use
            index_path: Path for persistent storage (used with DISK storage)
            storage_type: Whether to use in-memory or disk-based storage
            collection_name: Name for the collection
            db_manager: Optional DB manager for term/content lookups
            emotion_manager: Optional emotion manager for sentiment analysis

        Raises:
            InitializationError: If ChromaDB initialization fails
            ModelLoadError: If the embedding model fails to load
        """
        # Use provided values or fall back to configuration
        self.index_path = Path(index_path or config.vectorizer.index_path)
        self.storage_type = storage_type or config.vectorizer.storage_type
        self.db_manager = db_manager
        self.emotion_manager = emotion_manager
        self._has_persist_method = False
        self.model_name = model_name or config.vectorizer.model_name

        # Initialize embedding model
        try:
            self.model = SentenceTransformer(self.model_name)

            # Use dimension from config if not explicitly provided
            self.dimension = (
                dimension
                or config.vectorizer.dimension
                or self.model.get_sentence_embedding_dimension()
            )
            logging.info(
                f"Loaded model {self.model_name} with dimension {self.dimension}"
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load embedding model {self.model_name}: {e}"
            ) from e

        # Determine collection name from config or path if not provided
        if collection_name is None:
            collection_name = (
                config.vectorizer.collection_name
                or self.index_path.stem
                or "word_forge_vectors"
            )

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
            name=collection_name,
            metadata={
                "dimension": self.dimension,
                "model": self.model_name,
            },
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

    def format_with_instruction(
        self, text: str, template_key: str = "search", is_query: bool = True
    ) -> str:
        """
        Format text with appropriate instruction template for E5 model.

        Args:
            text: Raw text to format
            template_key: Key for selecting instruction template ("search", "definition", "similarity")
            is_query: Whether this text is a query (True) or document (False)

        Returns:
            Formatted text with appropriate instruction
        """
        # Get template from config
        template_dict = config.vectorizer.instruction_templates.get(
            template_key, config.vectorizer.instruction_templates["search"]
        )

        # Convert dict to InstructionTemplate for backward compatibility
        template = InstructionTemplate(
            task=template_dict["task"],
            query_prefix=template_dict["query_prefix"],
            document_prefix=template_dict["document_prefix"],
        )

        if is_query:
            # Format query with instruction
            return f"{template['query_prefix'].format(task=template['task'])}{text}"
        elif template["document_prefix"]:
            # Format document with prefix if one exists
            return f"{template['document_prefix']}{text}"
        else:
            # Most E5 document formats don't need a prefix
            return text

    def embed_text(
        self, text: str, template_key: str = "search", is_query: bool = True
    ) -> Vector:
        """
        Generate embedding vector for text using the E5 transformer model.

        Args:
            text: Text to embed
            template_key: Type of instruction to use
            is_query: Whether this is a query or document

        Returns:
            Numpy array containing the embedding vector

        Raises:
            ContentProcessingError: If embedding generation fails
        """
        try:
            # Format text with appropriate instruction if using E5 model
            if "e5" in self.model_name.lower():
                formatted_text = self.format_with_instruction(
                    text, template_key, is_query
                )
            else:
                formatted_text = text

            return self.model.encode(
                formatted_text, convert_to_numpy=True, normalize_embeddings=True
            )
        except Exception as e:
            raise ContentProcessingError(f"Failed to generate embedding: {e}") from e

    def _get_content_info(
        self, content_id: int, content_type: ContentType
    ) -> Dict[str, any]:
        """
        Get information about content based on its ID and type.

        Args:
            content_id: ID of the content
            content_type: Type of the content

        Returns:
            Dictionary with content information or empty dict if not found
        """
        if not self.db_manager:
            return {}

        try:
            # For word-related content
            if content_type in ("word", "definition", "example"):
                with sqlite3.connect(self.db_manager.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(SQL_GET_TERM_BY_ID, (content_id,))
                    result = cursor.fetchone()
                    if not result:
                        return {}
                    return {"term": result[0], "definition": result[1]}

            # For message content
            if content_type == "message":
                with sqlite3.connect(self.db_manager.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(SQL_GET_MESSAGE_TEXT, (content_id,))
                    result = cursor.fetchone()
                    if not result:
                        return {}
                    return {
                        "text": result[0],
                        "speaker": result[1],
                        "conversation_id": result[2],
                        "timestamp": result[3],
                    }

        except Exception as e:
            logging.warning(f"Error retrieving content info: {e}")

        return {}

    def _get_emotion_info(self, item_id: int) -> Dict[str, any]:
        """
        Get emotion information for a word.

        Args:
            item_id: ID of the word to look up

        Returns:
            Dictionary with emotion data or empty dict if not found/available
        """
        if not self.emotion_manager:
            return {}

        try:
            emotion_data = self.emotion_manager.get_word_emotion(item_id)
            if emotion_data:
                return {
                    "emotion_valence": emotion_data.get("valence"),
                    "emotion_arousal": emotion_data.get("arousal"),
                }
            return {}
        except Exception:
            return {}

    def process_word_entry(self, entry: WordEntryDict) -> List[Dict[str, any]]:
        """
        Process a word entry into multiple embeddings with metadata.

        Creates separate embeddings for:
        - The word itself
        - Its definition
        - Usage examples

        Args:
            entry: Complete word entry dictionary

        Returns:
            List of processed items with IDs, embeddings, and metadata

        Raises:
            ContentProcessingError: If processing fails
        """
        try:
            items = []
            word_id = entry["id"]
            term = entry["term"]
            definition = entry.get("definition", "")
            timestamp = entry.get("last_refreshed", 0.0)

            # Get emotion data if available
            emotion_data = self._get_emotion_info(word_id)

            # Process the word itself - use definition template for terms
            word_text = term
            word_embedding = self.embed_text(
                word_text, template_key="definition", is_query=True
            )
            word_metadata: VectorMetadata = {
                "original_id": word_id,
                "content_type": "word",
                "term": term,
                "timestamp": timestamp,
                **emotion_data,
            }
            items.append(
                {
                    "id": word_id,
                    "text": word_text,
                    "embedding": word_embedding,
                    "metadata": word_metadata,
                }
            )

            # Process the definition if present - use no instruction for documents
            if definition:
                def_id = word_id * 10000 + 1  # Unique ID for definition
                def_embedding = self.embed_text(definition, is_query=False)
                def_metadata = {
                    **word_metadata,
                    "content_type": "definition",
                    "definition": definition,
                }
                items.append(
                    {
                        "id": def_id,
                        "text": definition,
                        "embedding": def_embedding,
                        "metadata": def_metadata,
                    }
                )

            # Process usage examples
            examples = entry.get("usage_examples", [])
            for i, example in enumerate(examples):
                if not example:
                    continue

                example_id = word_id * 10000 + 100 + i  # Unique ID for each example
                example_embedding = self.embed_text(example, is_query=False)
                example_metadata = {**word_metadata, "content_type": "example"}
                items.append(
                    {
                        "id": example_id,
                        "text": example,
                        "embedding": example_embedding,
                        "metadata": example_metadata,
                    }
                )

            return items

        except Exception as e:
            raise ContentProcessingError(
                f"Failed to process word entry {entry['term']}: {e}"
            ) from e

    def store_word(self, entry: WordEntryDict) -> int:
        """
        Process and store a word entry with all its components.

        Args:
            entry: Complete word entry dictionary

        Returns:
            Number of vectors stored

        Raises:
            ContentProcessingError: If processing fails
        """
        try:
            # Process word into multiple items
            items = self.process_word_entry(entry)

            # Store each item
            for item in items:
                self.upsert(
                    vec_id=item["id"],
                    embedding=item["embedding"],
                    metadata=item["metadata"],
                    text=item["text"],
                )

            return len(items)

        except Exception as e:
            if isinstance(e, ContentProcessingError):
                raise
            raise ContentProcessingError(
                f"Failed to store word {entry['term']}: {e}"
            ) from e

    def upsert(
        self,
        vec_id: VectorID,
        embedding: Vector,
        metadata: Optional[VectorMetadata] = None,
        text: Optional[str] = None,
    ) -> None:
        """
        Add or update a vector in the store with metadata and optional text.

        Args:
            vec_id: Unique identifier for the vector
            embedding: Vector embedding to store
            metadata: Optional metadata for filtering and context
            text: Optional raw text for hybrid search

        Raises:
            DimensionMismatchError: If embedding dimension doesn't match expected dimension
            UpsertError: If ChromaDB operation fails
        """
        self._validate_vector_dimension(embedding, "Embedding")

        try:
            # ChromaDB expects string IDs and list embeddings
            str_id = str(vec_id)
            embedding_list = embedding.tolist()

            # Set minimum metadata if none provided
            if metadata is None:
                metadata = {"original_id": vec_id}

            # Add to ChromaDB
            if text:
                self.collection.upsert(
                    ids=[str_id],
                    embeddings=[embedding_list],
                    metadatas=[metadata],
                    documents=[text],
                )
            else:
                self.collection.upsert(
                    ids=[str_id], embeddings=[embedding_list], metadatas=[metadata]
                )

            # Persist changes for disk-based storage
            self._persist_if_needed()
        except Exception as e:
            raise UpsertError(f"Vector upsert failed: {str(e)}") from e

    def search(
        self,
        query_vector: Optional[Vector] = None,
        query_text: Optional[str] = None,
        k: int = 5,
        content_filters: Optional[Dict[str, any]] = None,
    ) -> List[SearchResultDict]:
        """
        Find k most similar vectors to the query.

        Args:
            query_vector: Query embedding to search against
            query_text: Raw text query (will be converted to vector if provided)
            k: Number of results to return (limited by collection size)
            content_filters: Optional metadata filters for the search

        Returns:
            List of result dictionaries containing id, distance and metadata,
            sorted by similarity (lowest distance first)

        Raises:
            DimensionMismatchError: If query vector dimension doesn't match expected dimension
            SearchError: If search operation fails
        """
        # Handle text query if provided
        if query_text and query_vector is None:
            query_vector = self.embed_text(query_text, is_query=True)

        if query_vector is None:
            raise ValueError("Must provide either query_vector or query_text")

        self._validate_vector_dimension(query_vector, "Query vector")

        # For backward compatibility with FAISS interface
        if self.collection.count() == 0:
            return []

        try:
            # Prepare filters if provided
            where_clause = content_filters if content_filters else None

            # Perform search with limit based on available items
            effective_k = min(k, self.collection.count())
            results = self.collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=effective_k,
                where=where_clause,
            )

            # No results case
            if not results["ids"][0]:
                return []

            # Process results
            raw_ids = results["ids"][0]
            distances = self._convert_similarities_to_distances(results["distances"][0])
            metadatas = results["metadatas"][0] if "metadatas" in results else None
            documents = results["documents"][0] if "documents" in results else None

            # Build result dictionaries
            search_results: List[SearchResultDict] = []
            for i, id_str in enumerate(raw_ids):
                result_id = int(id_str)
                result_dict: SearchResultDict = {
                    "id": result_id,
                    "distance": distances[i],
                    "metadata": (
                        cast(VectorMetadata, metadatas[i]) if metadatas else None
                    ),
                    "text": documents[i] if documents and i < len(documents) else None,
                }
                search_results.append(result_dict)

            return search_results
        except Exception as e:
            raise SearchError(f"Vector search failed: {str(e)}") from e

    def get_legacy_search_results(
        self, query_vector: Vector, k: int = 5
    ) -> SearchResult:
        """
        Perform search with legacy return format (list of ID-distance tuples).

        Args:
            query_vector: Query embedding to search against
            k: Number of results to return

        Returns:
            List of (id, distance) tuples sorted by similarity

        Raises:
            Same exceptions as search()
        """
        results = self.search(query_vector=query_vector, k=k)
        return [(item["id"], item["distance"]) for item in results]

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


def main() -> None:
    """
    Demonstrate the Multilingual-E5 vector store capabilities.
    """
    from word_forge.database.db_manager import DBManager
    from word_forge.emotion.emotion_manager import EmotionManager

    print(
        "Initializing Word Forge vector store with Multilingual-E5-large-instruct model..."
    )

    # Create test database and managers
    db_path = "word_forge.sqlite"
    db_manager = DBManager(db_path=db_path)
    emotion_manager = EmotionManager(db_manager)

    # Initialize vector store with multilingual capabilities
    vector_store = VectorStore(
        model_name="intfloat/multilingual-e5-large-instruct",
        storage_type=StorageType.MEMORY,
        db_manager=db_manager,
        emotion_manager=emotion_manager,
    )

    print(
        f"Vector store initialized with {vector_store.dimension}-dimensional embeddings"
    )
    print(f"Using model: {vector_store.model_name}")

    # Create multilingual test content
    test_words = [
        ("algorithm", "A step-by-step procedure for solving a problem", "en"),
        ("pumpkin", "A large orange fruit with a thick shell", "en"),
        ("南瓜", "一种橙色的大型水果，常用于食品和装饰", "zh"),  # Chinese: pumpkin
        (
            "recette",
            "Instructions pour préparer un plat culinaire",
            "fr",
        ),  # French: recipe
    ]

    # Store multilingual words
    print("\n=== Storing Multilingual Content ===")
    for term, definition, language in test_words:
        try:
            # Create or update word entry
            db_manager.insert_or_update_word(
                term=term, definition=definition, part_of_speech="noun"
            )

            # Get word entry and add language metadata
            entry = db_manager.get_word_entry(term)

            # Process and store the word
            vector_store.store_word(entry)
            print(f"Stored '{term}' ({language})")

        except Exception as e:
            print(f"Error processing '{term}': {e}")

    # Test multilingual queries
    print("\n=== Multilingual Search ===")
    queries = [
        ("How to cook pumpkin?", "en"),
        ("南瓜的做法", "zh"),  # How to prepare pumpkin
        ("recette de cuisine", "fr"),  # Cooking recipe
    ]

    for query_text, language in queries:
        print(f"\nSearching for: '{query_text}' ({language})")

        # Search using the query text directly
        results = vector_store.search(query_text=query_text, k=2)

        # Display results
        if results:
            for i, result in enumerate(results):
                term = (
                    result["metadata"].get("term", "Unknown")
                    if result["metadata"]
                    else "Unknown"
                )
                distance = result["distance"]
                print(f"  Result {i+1}: {term}, Distance: {distance:.4f}")
                if result["text"]:
                    print(f"  Text: {result['text']}")
        else:
            print("  No results found")

    # Demonstrate cross-lingual retrieval
    print("\n=== Cross-lingual Retrieval Example ===")
    english_query = "pumpkin recipes"
    chinese_results = vector_store.search(query_text=english_query, k=2)

    print(f"Query '{english_query}' results:")
    for i, result in enumerate(chinese_results):
        term = (
            result["metadata"].get("term", "Unknown")
            if result["metadata"]
            else "Unknown"
        )
        print(f"  Result {i+1}: {term}, Distance: {result['distance']:.4f}")

    print("\nDemo completed!")


if __name__ == "__main__":
    main()
