"""
Vector Store module for Word Forge.

This module provides storage and retrieval capabilities for vector embeddings,
enabling semantic search and similarity operations across linguistic data.
It supports multiple storage backends and embedding models with configurable
parameters for balancing performance and accuracy.

Architecture:
    ┌─────────────────┐
    │   VectorStore   │
    └────────┬────────┘
             │
    ┌────────┼────────┐
    │    Components   │
    └─────────────────┘
    ┌─────┬─────┬─────┐
    │Model│Index│Query│
    └─────┴─────┴─────┘
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
    cast,
    overload,
)

import chromadb
import numpy as np
import torch
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from word_forge.config import config
from word_forge.database.db_manager import DatabaseError, DBManager, WordEntryDict
from word_forge.emotion.emotion_manager import EmotionManager

# Type definitions for clarity and constraint
VectorID = int  # Unique identifier for vectors
SearchResult = List[Tuple[VectorID, float]]  # (id, distance) pairs
ContentType = Literal["word", "definition", "example", "message", "conversation"]
EmbeddingList = List[float]  # Type for ChromaDB's embedding format
QueryType = Literal["search", "definition", "similarity"]
TemplateDict = Dict[str, Optional[str]]
WordID = Union[int, str]  # ID can be int or str for ChromaDB compatibility


class VectorMetadata(TypedDict, total=False):
    """
    Metadata associated with stored vectors.

    Structured information that accompanies vector embeddings to provide
    context and support filtering operations during search.

    Attributes:
        original_id: Source entity identifier
        content_type: Category of content this vector represents
        term: Word or phrase if this vector represents lexical content
        definition: Meaning of the term if applicable
        speaker: Person who created this content if from a conversation
        emotion_valence: Emotional valence score if sentiment analyzed
        emotion_arousal: Emotional arousal intensity if sentiment analyzed
        emotion_label: Text label for the dominant emotion
        conversation_id: Parent conversation identifier for message content
        timestamp: When this content was created or processed
        language: Language code for the content
    """

    original_id: int
    content_type: ContentType
    term: Optional[str]
    definition: Optional[str]
    speaker: Optional[str]
    emotion_valence: Optional[float]
    emotion_arousal: Optional[float]
    emotion_label: Optional[str]
    conversation_id: Optional[int]
    timestamp: Optional[float]
    language: Optional[str]


class SearchResultDict(TypedDict):
    """
    Type definition for search result items.

    Structured format for returning vector search results with all
    relevant metadata and context.

    Attributes:
        id: int - Unique identifier of the matching item
        distance: float - Semantic distance from query (lower is better)
        metadata: Optional[VectorMetadata] - Associated metadata for the match
        text: Optional[str] - Raw text content if available
    """

    id: int
    distance: float
    metadata: Optional[VectorMetadata]
    text: Optional[str]


class InstructionTemplate(TypedDict):
    """
    Type definition for instruction template.

    Format specification for instruction-tuned language models
    that require specific prompting patterns.

    Attributes:
        task: Description of the task to perform
        query_prefix: Text to prepend to query inputs
        document_prefix: Optional text to prepend to documents
    """

    task: str
    query_prefix: str
    document_prefix: Optional[str]


class ChromaCollection(Protocol):
    """
    Protocol defining required ChromaDB collection interface.

    Abstract interface that ensures compatibility with the ChromaDB
    collection API regardless of implementation details.
    """

    def count(self) -> int:
        """Return the number of items in the collection."""
        ...

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[str]] = None,
    ) -> None:
        """Insert or update items in the collection."""
        ...

    def query(
        self,
        query_embeddings: Optional[List[List[float]]] = None,
        query_texts: Optional[List[str]] = None,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Any]]:
        """Query the collection for similar items."""
        ...

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete items from the collection by ID or filter."""
        ...


class ChromaClient(Protocol):
    """
    Protocol defining required ChromaDB client interface.

    Abstract interface that ensures compatibility with the ChromaDB
    client API regardless of implementation details.
    """

    def get_or_create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ChromaCollection:
        """Get or create a collection with the given name."""
        ...

    def persist(self) -> None:
        """Persist the database to disk."""
        ...


# Using StorageType from centralized config
StorageType = config.vectorizer.storage_type.__class__


class VectorStoreError(DatabaseError):
    """Base exception for vector store operations."""

    pass


class InitializationError(VectorStoreError):
    """
    Raised when vector store initialization fails.

    This occurs when the vector store cannot be properly initialized,
    such as when the embedding model fails to load or the storage
    backend cannot be configured.
    """

    pass


class ModelLoadError(VectorStoreError):
    """
    Raised when embedding model loading fails.

    This occurs when the specified embedding model cannot be loaded,
    such as when the model file is missing or corrupted.
    """

    pass


class UpsertError(VectorStoreError):
    """
    Raised when adding or updating vectors fails.

    This occurs when an attempt to store or update vectors in the
    database fails, such as due to invalid data or storage constraints.
    """

    pass


class SearchError(VectorStoreError):
    """
    Raised when vector similarity search fails.

    This occurs when a semantic search operation cannot be completed,
    such as due to missing or incompatible data.
    """

    pass


class DimensionMismatchError(VectorStoreError):
    """
    Raised when vector dimensions don't match expected dimensions.

    This occurs when the dimension of a provided vector doesn't match
    the expected dimension of the vector store, which would lead to
    incompatible operations.
    """

    pass


class ContentProcessingError(VectorStoreError):
    """
    Raised when processing content for embedding fails.

    This occurs when text content cannot be properly processed into
    vector embeddings, such as due to invalid format or content issues.
    """

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

    Attributes:
        model: SentenceTransformer model for generating embeddings
        dimension: Vector dimension size (typically 1024 for E5 models)
        model_name: Name of the embedding model being used
        client: ChromaDB client for vector storage operations
        collection: ChromaDB collection storing the vectors
        index_path: Path to persistent storage location
        storage_type: Whether using memory or disk storage
        db_manager: Optional database manager for content lookups
        emotion_manager: Optional emotion manager for sentiment analysis
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
            # Cast to str to satisfy type checker
            model_name_str: str = str(self.model_name)
            self.model = SentenceTransformer(model_name_str)  # type: ignore
        except Exception as e:
            raise ModelLoadError(f"Failed to load embedding model: {str(e)}") from e

        # Determine embedding dimension - Fixed the error in dimension handling
        embedding_dim = 0
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            # Handle both function and property cases
            if callable(self.model.get_sentence_embedding_dimension):
                embedding_dim = self.model.get_sentence_embedding_dimension()
            else:
                embedding_dim = self.model.get_sentence_embedding_dimension

            # Convert tensor to scalar if needed
            if hasattr(embedding_dim, "item"):
                embedding_dim = embedding_dim.item()

        # Safe fallback chain for dimension
        if dimension is not None:
            self.dimension = dimension
        elif config.vectorizer.dimension is not None:
            self.dimension = config.vectorizer.dimension
        elif embedding_dim > 0:
            self.dimension = embedding_dim
        else:
            # Default fallback
            self.dimension = 1024

        logging.info(f"Loaded model {self.model_name} with dimension {self.dimension}")

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

        # Ensure directory exists for DISK storage
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
        self, vector: NDArray[np.float32], context: str = "Vector"
    ) -> None:
        """Validate that vector dimensions match expected dimensions."""
        if len(vector.shape) != 1 or vector.shape[0] != self.dimension:
            raise DimensionMismatchError(
                f"{context} dimension {vector.shape[0]} doesn't match expected {self.dimension}"
            )

    def format_with_instruction(
        self, text: str, template_key: str = "search", is_query: bool = True
    ) -> str:
        """
        Format text with appropriate instruction template for E5 model.

        Applies instruction-tuning conventions to raw text, enabling the model
        to understand the context and intent of the embedding operation.

        Args:
            text: Raw text to format
            template_key: Key for selecting instruction template ("search", "definition", "similarity")
            is_query: Whether this text is a query (True) or document (False)

        Returns:
            Formatted text with appropriate instruction
        """
        # Define default template with optimal structure for E5 models
        default_template: InstructionTemplate = {
            "task": "Search for relevant information",
            "query_prefix": "Instruct: {task}\nQuery: ",
            "document_prefix": None,
        }

        # Access templates with robust fallback chain
        if hasattr(config.vectorizer, "instruction_templates"):
            templates = config.vectorizer.instruction_templates
            if template_key in templates:
                template_dict = cast(InstructionTemplate, templates[template_key])
            else:
                template_dict = cast(
                    InstructionTemplate, templates.get("search", default_template)
                )
        else:
            template_dict = default_template

        # Type-safe extraction with fallbacks for each field
        task = template_dict.get("task") or default_template["task"]
        query_prefix = (
            template_dict.get("query_prefix") or default_template["query_prefix"]
        )
        document_prefix = template_dict.get("document_prefix")

        # Apply formatting based on content type
        if is_query:
            return f"{query_prefix.format(task=task)}{text}"
        elif document_prefix:
            return f"{document_prefix}{text}"
        else:
            # Standard document formatting for E5 models
            return text

    def embed_text(
        self, text: str, template_key: str = "search", is_query: bool = True
    ) -> NDArray[np.float32]:
        """
        Embed text using the SentenceTransformer model.

        Transforms raw text into a high-dimensional vector representation
        while applying appropriate instruction formatting based on context.

        Args:
            text: Text to embed
            template_key: Key for instruction template selection
            is_query: Whether this is a query (affects instruction formatting)

        Returns:
            Embedding vector as numpy array (normalized by default)

        Raises:
            ContentProcessingError: If embedding fails due to model issues or invalid input
        """
        if not text or not text.strip():
            raise ContentProcessingError("Cannot embed empty text")

        try:
            # Apply instruction formatting
            formatted_text = self.format_with_instruction(text, template_key, is_query)

            # Generate embedding with optimal parameters for similarity search
            vector = self.model.encode(  # type: ignore
                sentences=formatted_text,  # Use the formatted text directly as sentences
                batch_size=10,  # Batch text optimization
                convert_to_numpy=True,
                normalize_embeddings=True,  # Pre-normalize for cosine similarity
                show_progress_bar=True,  # Show Progress
                output_value="sentence_embedding",  # Ensure correct output
                precision="float32",  # Use float32 for consistency
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Ensure correct type for consistent operations
            vector = vector.astype(np.float32)

            # Validate dimensions match expected size
            self._validate_vector_dimension(
                vector,
                f"Embedding for '{text[:self.dimension]}{'...' if len(text) > self.dimension else ''}'",
            )

            return vector
        except Exception as e:
            raise ContentProcessingError(f"Failed to embed text: {str(e)}") from e

    def _get_content_info(
        self, content_id: int, content_type: ContentType
    ) -> Dict[str, Any]:
        """
        Get information about content based on its ID and type.

        Retrieves detailed information for various content types from the database,
        ensuring type consistency and error resilience.

        Args:
            content_id: ID of the content
            content_type: Type of the content ("word", "definition", "example", "message", "conversation")

        Returns:
            Dictionary with content information or empty dict if not found/accessible
        """
        if not self.db_manager:
            return {}

        try:
            if content_type in ("word", "definition", "example"):
                # Get word information - fixed int/str type mismatch
                word = self.db_manager.get_word_entry(str(content_id))
                if word:
                    return {
                        "term": word.get("term", ""),
                        "definition": word.get("definition", ""),
                        "language": word.get(
                            "language", "en"
                        ),  # Add language awareness
                    }
            elif content_type in ("message", "conversation"):
                # Get message information with proper error handling
                results = self.db_manager.execute_query(
                    SQL_GET_MESSAGE_TEXT, (content_id,)
                )
                if results and len(results) > 0:
                    message_data = results[0]
                    return {
                        "text": message_data[0] if len(message_data) > 0 else "",
                        "speaker": message_data[1] if len(message_data) > 1 else None,
                        "conversation_id": (
                            message_data[2] if len(message_data) > 2 else None
                        ),
                        "timestamp": message_data[3] if len(message_data) > 3 else None,
                    }
        except Exception as e:
            logging.warning(
                f"Failed to get content info for {content_type} {content_id}: {str(e)}"
            )

        return {}

    def _get_emotion_info(self, item_id: int) -> Dict[str, Any]:
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
                    "emotion_label": emotion_data.get("label"),
                }
        except Exception:
            # Silently fail if emotion data can't be retrieved
            pass

        return {}

    def process_word_entry(self, entry: WordEntryDict) -> List[Dict[str, Any]]:
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
            word_id = entry["id"]
            term = entry["term"]
            definition = entry["definition"]
            usage_examples_input = entry.get("usage_examples", "")
            # Handle both cases: if input is already a list or if it's a string that needs parsing
            if usage_examples_input:
                examples = usage_examples_input
            else:
                examples = self._parse_usage_examples(str(usage_examples_input))

            # Get emotion data if available
            emotion_data = self._get_emotion_info(word_id)

            # Prepare results list with type safety
            items: List[Dict[str, Any]] = []

            # Process term
            term_text = f"{term} - {definition}"
            word_embedding = self.embed_text(term_text, "definition", is_query=False)

            # Create metadata with proper type alignment
            word_metadata: Dict[str, Any] = {
                "original_id": word_id,
                "content_type": "word",
                "term": term,
                "definition": definition,
            }
            # Add emotion data if available
            for k, v in emotion_data.items():
                word_metadata[k] = v

            items.append(
                {
                    "id": f"w_{word_id}",
                    "vector": word_embedding,
                    "metadata": word_metadata,
                    "text": term_text,
                }
            )

            # Process definition separately
            def_embedding = self.embed_text(definition, "definition", is_query=False)
            # Create metadata with proper type alignment
            def_metadata: Dict[str, Any] = {
                "original_id": word_id,
                "content_type": "definition",
                "term": term,
                "definition": definition,
            }
            # Add emotion data if available
            for k, v in emotion_data.items():
                def_metadata[k] = v

            items.append(
                {
                    "id": f"d_{word_id}",
                    "vector": def_embedding,
                    "metadata": def_metadata,
                    "text": definition,
                }
            )

            # Process each example
            for i, example in enumerate(examples):
                if not example.strip():
                    continue

                example_embedding = self.embed_text(
                    example, "similarity", is_query=False
                )
                # Create metadata with proper type alignment
                example_metadata: Dict[str, Any] = {
                    "original_id": word_id,
                    "content_type": "example",
                    "term": term,
                }
                # Add emotion data if available
                for k, v in emotion_data.items():
                    example_metadata[k] = v

                items.append(
                    {
                        "id": f"e_{word_id}_{i}",
                        "vector": example_embedding,
                        "metadata": example_metadata,
                        "text": example,
                    }
                )

            return items

        except Exception as e:
            raise ContentProcessingError(
                f"Failed to process word entry: {str(e)}"
            ) from e

    def _parse_usage_examples(self, examples_string: str) -> List[str]:
        """
        Parse a string of semicolon-separated examples into a list.

        Args:
            examples_string: String containing examples separated by semicolons

        Returns:
            List of individual usage examples
        """
        if not examples_string:
            return []
        return [ex.strip() for ex in examples_string.split(";") if ex.strip()]

    def store_word(self, entry: WordEntryDict) -> int:
        """
        Process and store a word entry with all its components.

        Creates vector embeddings for the word term, definition, and usage examples,
        then stores them in the vector database with appropriate metadata.

        Args:
            entry: Complete word entry dictionary containing term, definition, and examples

        Returns:
            int: Number of vectors successfully stored

        Raises:
            ContentProcessingError: If processing or storage fails
        """
        try:
            items = self.process_word_entry(entry)

            for item in items:
                # Extract vector_id from the id string (e.g., "w_123" -> 123)
                parts = item["id"].split("_")
                vec_id = int(parts[1])

                # Call upsert with extracted ID and other components
                self.upsert(
                    vec_id=vec_id,
                    embedding=item["vector"],
                    metadata=item["metadata"],
                    text=item["text"],
                )

            return len(items)

        except Exception as e:
            raise ContentProcessingError(f"Failed to store word: {str(e)}") from e

    def delete_vectors_for_word(self, word_id: WordID) -> int:
        """
        Delete all vectors associated with a specific word.

        Removes vectors associated with the word from the vector store,
        enabling clean updates and replacements without duplicate entries.

        Args:
            word_id: ID of the word whose vectors should be deleted

        Returns:
            Number of vectors deleted

        Raises:
            VectorStoreError: If the deletion operation fails

        Examples:
            >>> store = VectorStore(dimension=384)
            >>> # After adding vectors for a word
            >>> deleted_count = store.delete_vectors_for_word(123)
            >>> print(f"Deleted {deleted_count} vectors")
        """
        try:
            # Convert word_id to string for ChromaDB
            word_id_str = str(word_id)

            # Use metadata filter to find all vectors for this word
            where_filter = {"original_id": word_id}

            # Try to delete vectors by ID and filter
            deleted_count = 0

            # First try direct ID-based deletion
            try:
                self.collection.delete(ids=[word_id_str])
                deleted_count += 1
            except Exception as direct_error:
                logging.debug(f"Direct ID deletion failed: {str(direct_error)}")

            # Then try filter-based deletion if supported
            try:
                self.collection.delete(where=where_filter)
                # We don't know exactly how many were deleted, but at least 1
                deleted_count = max(deleted_count, 1)
            except Exception as filter_error:
                logging.debug(f"Filter-based deletion failed: {str(filter_error)}")

            # If both approaches failed, propagate the error
            if deleted_count == 0:
                raise VectorStoreError("Neither ID nor filter-based deletion succeeded")

            # Persist changes if using disk storage
            self._persist_if_needed()

            return deleted_count

        except Exception as e:
            raise VectorStoreError(
                f"Failed to delete vectors for word {word_id}: {str(e)}"
            ) from e

    def upsert(
        self,
        vec_id: VectorID,
        embedding: NDArray[np.float32],
        metadata: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
    ) -> None:
        """
        Add or update a vector in the store with metadata and optional text.

        This function validates the vector dimension, sanitizes metadata to ensure
        compatibility with ChromaDB's requirements, and handles the storage operation.

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
            # Convert embedding to list for Chroma
            embedding_list = embedding.tolist()

            # Convert ID to string (ChromaDB requirement)
            id_str = str(vec_id)

            # Sanitize metadata - ChromaDB only accepts primitives
            sanitized_metadata = self._sanitize_metadata(metadata or {})

            # Upsert into collection
            self.collection.upsert(
                ids=[id_str],
                embeddings=[embedding_list],
                metadatas=[sanitized_metadata] if sanitized_metadata else None,
                documents=[text] if text else None,
            )

            # Persist if using disk storage
            self._persist_if_needed()

        except Exception as e:
            raise UpsertError(f"Failed to store vector: {str(e)}") from e

    def _sanitize_metadata(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Union[str, int, float, bool]]:
        """
        Sanitize metadata to ensure compatibility with ChromaDB requirements.

        ChromaDB only accepts primitive types (str, int, float, bool) as metadata values.
        This function converts None values to empty strings and ensures all values are
        compatible types.

        Args:
            metadata: Raw metadata dictionary with potential None values

        Returns:
            Dict[str, Union[str, int, float, bool]]: Sanitized metadata with compatible types
        """
        sanitized: Dict[str, Union[str, int, float, bool]] = {}

        for key, value in metadata.items():
            if value is None:
                # Convert None to empty string for ChromaDB compatibility
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                # Keep primitive types as-is
                sanitized[key] = value
            else:
                # Convert other types to string representation
                sanitized[key] = str(value)

        return sanitized

    @overload
    def search(
        self,
        query_vector: NDArray[np.float32],
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResultDict]: ...

    @overload
    def search(
        self,
        *,
        query_text: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResultDict]: ...

    def search(
        self,
        query_vector: Optional[NDArray[np.float32]] = None,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        *,
        query_text: Optional[str] = None,
    ) -> List[SearchResultDict]:
        """
        Search for semantically similar vectors using vector or text query.

        Performs a similarity search against the vector database, finding items
        that are semantically closest to the provided query. Results are sorted
        by similarity (closest first). This method supports two search modes:
        1. Direct vector search - when you already have an embedding
        2. Text-based search - automatically converts text to embedding

        Args:
            query_vector: Vector representation to search against
                (mutually exclusive with query_text)
            query_text: Text to search for (will be converted to a vector)
            k: Maximum number of results to return (default: 5)
            filter_metadata: Optional metadata filters to restrict search
                (e.g., {"language": "en"})

        Returns:
            List[SearchResultDict]: Search results with metadata, sorted by similarity
                Each result contains:
                - id: Unique identifier of the matching item
                - distance: Semantic distance from query (lower is better)
                - metadata: Associated metadata for the match
                - text: Raw text content if available

        Raises:
            SearchError: If search parameters are invalid or operation fails
            ContentProcessingError: If embedding query_text fails
            DimensionMismatchError: If query vector dimensions are incorrect
            ValueError: If k is not a positive integer

        Examples:
            >>> # Search by text with language filter
            >>> results = vector_store.search(
            ...     query_text="quantum computing",
            ...     k=5,
            ...     filter_metadata={"language": "en"}
            ... )
            >>> print(f"Found {len(results)} matches")

            >>> # Search by vector
            >>> results = vector_store.search(query_vector=embedding)
        """
        # Validate k parameter
        if not k or k <= 0:
            raise ValueError(f"Parameter 'k' must be a positive integer, got {k}")

        # Input validation - exclusive parameters
        if query_vector is None and query_text is None:
            raise SearchError(
                "Search requires either query_vector or query_text parameter"
            )

        if query_vector is not None and query_text is not None:
            raise SearchError("Provide either query_vector or query_text, not both")

        # Prepare search vector
        search_vector = self._prepare_search_vector(query_vector, query_text)

        # Execute search and return results
        return self._execute_vector_search(
            search_vector=search_vector, k=k, filter_metadata=filter_metadata
        )

    def _prepare_search_vector(
        self, query_vector: Optional[NDArray[np.float32]], query_text: Optional[str]
    ) -> NDArray[np.float32]:
        """
        Prepare the search vector from either direct vector or text input.

        Args:
            query_vector: Pre-computed vector embedding if available
            query_text: Text to embed if vector not provided

        Returns:
            NDArray[np.float32]: Vector to use for similarity search

        Raises:
            ContentProcessingError: If text embedding fails
            DimensionMismatchError: If vector has incorrect dimensions
        """
        # Case 1: Direct vector provided
        if query_vector is not None:
            self._validate_vector_dimension(query_vector, context="Query vector")
            return query_vector

        # Case 2: Text query provided
        assert query_text is not None, "Both query_vector and query_text cannot be None"

        try:
            embedded_vector = self.embed_text(
                query_text, template_key="search", is_query=True
            )
            self._validate_vector_dimension(embedded_vector, context="Embedded query")
            return embedded_vector
        except Exception as e:
            # Preserve dimension mismatch errors but wrap others
            if isinstance(e, DimensionMismatchError):
                raise e
            raise ContentProcessingError(
                f"Failed to embed query text '{query_text[:50]}...': {str(e)}"
            ) from e

    def _execute_vector_search(
        self,
        search_vector: NDArray[np.float32],
        k: int,
        filter_metadata: Optional[Dict[str, Any]],
    ) -> List[SearchResultDict]:
        """
        Execute similarity search against the vector database.

        Args:
            search_vector: Vector to search against
            k: Maximum number of results to return
            filter_metadata: Optional filters to apply to search

        Returns:
            List[SearchResultDict]: Search results sorted by similarity

        Raises:
            SearchError: If the search operation fails
        """
        try:
            # Ensure k is reasonable to prevent performance issues
            max_allowed = getattr(config.vectorizer, "max_results", 100)
            adjusted_k = max(1, min(k, max_allowed))

            # Performance logging for large searches
            if adjusted_k != k:
                logging.warning(
                    f"Requested {k} results but limited to {adjusted_k} for performance"
                )

            # Record search operation start time
            start_time = time.time()

            # Execute query against ChromaDB collection
            query_results = self.collection.query(
                query_embeddings=[search_vector.tolist()],
                n_results=adjusted_k,
                where=filter_metadata,  # ChromaDB accepts None as "no filter"
            )

            # Log slow searches for performance monitoring
            search_time = time.time() - start_time
            if search_time > 1.0:
                logging.info(
                    f"Vector search took {search_time:.2f}s for k={adjusted_k}, "
                    f"filters={filter_metadata}"
                )

            # Process results into standardized format
            return self._process_chromadb_results(query_results)
        except Exception as e:
            error_msg = f"Vector search failed: {str(e)}"
            logging.error(
                f"{error_msg} | Vector shape: {search_vector.shape} | "
                f"Filters: {filter_metadata} | Collection size: {self.collection.count()}"
            )
            raise SearchError(error_msg) from e

    def _process_chromadb_results(
        self, query_results: Dict[str, List[Any]]
    ) -> List[SearchResultDict]:
        """
        Process raw ChromaDB results into standardized search result format.

        Args:
            query_results: Raw results from ChromaDB query operation

        Returns:
            List[SearchResultDict]: Standardized search results

        Raises:
            SearchError: If results cannot be properly processed
        """
        try:
            # Handle empty results gracefully
            if (
                not query_results
                or "ids" not in query_results
                or not query_results["ids"]
            ):
                return []

            # Access first query's results (ChromaDB uses nested arrays)
            ids_list: List[str] = (
                query_results["ids"][0] if query_results["ids"] else []
            )
            if not ids_list:
                return []

            # Convert string IDs to integers
            ids = [int(id_str) for id_str in ids_list]

            # Safely extract and process distances
            distances_data = query_results.get("distances", [[]])
            distances_list: List[float] = distances_data[0] if distances_data else []
            distances = self._convert_similarities_to_distances(distances_list)

            # Safely extract metadata and documents with proper defaults
            metadatas_data = query_results.get("metadatas", [[]])
            metadatas = metadatas_data[0] if metadatas_data else [None] * len(ids)

            documents_data = query_results.get("documents", [[]])
            documents = documents_data[0] if documents_data else [None] * len(ids)

            # Build result objects with proper type safety
            results: List[SearchResultDict] = []
            for i, id_ in enumerate(ids):
                # Get corresponding values with bounds checking
                distance = (
                    distances[i] if i < len(distances) else 2.0
                )  # Default to max distance
                metadata = metadatas[i] if i < len(metadatas) else None
                document = documents[i] if i < len(documents) else None

                # Construct the result with proper typing
                result = SearchResultDict(
                    id=id_,
                    distance=distance,
                    metadata=cast(Optional[VectorMetadata], metadata),
                    text=document,
                )
                results.append(result)

            return results
        except Exception as e:
            raise SearchError(f"Failed to process search results: {str(e)}") from e

    def get_legacy_search_results(
        self, query_vector: NDArray[np.float32], k: int = 5
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

        ChromaDB returns similarity scores in range [0,1] where 1.0 represents
        a perfect match. For compatibility with FAISS and other distance-based
        systems, this method converts to a distance metric where 0.0 represents
        a perfect match.

        Args:
            similarities: List of similarity scores from ChromaDB

        Returns:
            List of distance scores (1.0 - similarity) with identical ordering
        """
        return [1.0 - similarity for similarity in similarities]

    def _persist_if_needed(self) -> None:
        """
        Persist vector store to disk when appropriate.

        Saves the current state to persistent storage if:
        1. Using disk-based storage (not in-memory)
        2. Client implements the persist() method

        This safely handles various client implementations and prevents
        runtime errors when the persist method isn't available.
        """
        if self.storage_type == StorageType.DISK and self._has_persist_method:
            try:
                self.client.persist()
            except Exception as e:
                logging.warning(
                    f"Failed to persist vector store to {self.index_path}: {str(e)}"
                )

    def __del__(self) -> None:
        """
        Finalize object destruction with proper resource cleanup.

        Ensures vector data is persisted before the object is destroyed
        by the garbage collector, preventing data loss during shutdown
        or scope exit.

        Exceptions are silently handled to avoid crashes during garbage
        collection, as per Python's recommendation for __del__ methods.
        """
        try:
            self._persist_if_needed()
        except Exception:
            # Silent handling during garbage collection is appropriate
            # as __del__ should never raise exceptions
            pass
