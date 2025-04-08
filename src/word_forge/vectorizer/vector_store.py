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
import os
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
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from word_forge.config import config
from word_forge.database.database_manager import DatabaseError, DBManager, WordEntryDict
from word_forge.emotion.emotion_manager import EmotionManager

# Type definitions for clarity and constraint
VectorID = Union[int, str]  # Unique identifier for vectors (compatible with ChromaDB)
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


class VectorItem(TypedDict):
    """
    Type definition for vector items to be stored.

    Structured format for vector data with associated metadata and text.

    Attributes:
        id: VectorID - Unique identifier for the vector
        text: str - Text content associated with the vector
        metadata: VectorMetadata - Metadata for filtering and context
        vector: NDArray[np.float32] - Vector embedding
    """

    id: VectorID
    text: str
    metadata: VectorMetadata
    vector: NDArray[np.float32]


class InstructionTemplate(TypedDict):
    """
    Type definition for instruction template.

    Format specification for instruction-tuned language models
    that require specific prompting patterns.

    Attributes:
        task: str - Description of the task to perform
        query_prefix: str - Text to prepend to query inputs
        document_prefix: Optional[str] - Optional text to prepend to documents
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

    # Declare class attributes with type annotations
    dimension: int
    model: SentenceTransformer
    model_name: str
    client: ChromaClient
    collection: ChromaCollection
    index_path: Path
    storage_type: StorageType
    db_manager: Optional[DBManager]
    emotion_manager: Optional[EmotionManager]
    logger: logging.Logger
    instruction_templates: Dict[str, InstructionTemplate]

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

        Sets up the embedding model, storage backend, and all necessary
        components for the vector store to function properly. Connects to
        persistent storage and loads or initializes required models.

        Args:
            dimension: Optional override for vector dimensions
            model_name: Optional embedding model to use
            index_path: Optional path for vector storage
            storage_type: Optional storage type (memory or disk)
            collection_name: Optional name for the vector collection
            db_manager: Optional database manager for content lookup
            emotion_manager: Optional emotion manager for sentiment analysis

        Raises:
            InitializationError: If initialization fails
            ModelLoadError: If embedding model cannot be loaded
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)

        # Initialize dimension with a default value
        self.dimension = 0

        # Store configuration
        self.index_path = Path(index_path or config.vectorizer.index_path)
        self.storage_type = storage_type or config.vectorizer.storage_type
        self.db_manager = db_manager
        self.emotion_manager = emotion_manager
        self.model_name = model_name or config.vectorizer.model_name

        # Validate and create storage directory if needed
        if self.storage_type != StorageType.MEMORY:
            os.makedirs(self.index_path, exist_ok=True)

        try:
            if not hasattr(self, "model"):
                self.model = SentenceTransformer(self.model_name)  # type: ignore
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load embedding model '{self.model_name}': {str(e)}"
            ) from e

        # Initialize embedding model first to determine dimension
        try:
            if dimension is not None:
                self.dimension = dimension
            elif (
                hasattr(config.vectorizer, "dimension")
                and config.vectorizer.dimension is not None
            ):
                self.dimension = config.vectorizer.dimension
            # If model wasn't loaded to get dimension, load it now
        except Exception as e:
            raise ModelLoadError(
                f"Failed to determine vector dimension: {str(e)}"
            ) from e

        # Initialize ChromaDB client and collection
        try:
            self.client = self._create_client()
            collection_name = collection_name or (
                config.vectorizer.collection_name or "word_forge_vectors"
            )
            self.collection = self._initialize_collection(collection_name)
        except Exception as e:
            raise InitializationError(
                f"Failed to initialize vector store: {str(e)}"
            ) from e
        # Load instruction templates if available
        self.instruction_templates: Dict[str, InstructionTemplate] = {}
        if hasattr(config.vectorizer, "instruction_templates"):
            # Convert the templates to the expected type
            for key, template in config.vectorizer.instruction_templates.items():
                # Cast QueryType key to string
                str_key = str(key)
                # Convert TemplateDict to InstructionTemplate
                self.instruction_templates[str_key] = cast(
                    InstructionTemplate, template
                )

        self.logger.info(
            f"VectorStore initialized: model={self.model_name}, "
            f"dimension={self.dimension}, storage={self.storage_type.name.lower()}, "
            f"path={self.index_path}"
        )

    def _create_client(self) -> ChromaClient:
        """
        Create an appropriate ChromaDB client based on configuration.

        Returns:
            ChromaClient: Configured ChromaDB client for vector operations

        Raises:
            InitializationError: If client creation fails
        """
        try:
            if self.storage_type == StorageType.MEMORY:
                return cast(ChromaClient, chromadb.Client())
            else:
                return cast(
                    ChromaClient, chromadb.PersistentClient(path=str(self.index_path))
                )
        except Exception as e:
            raise InitializationError(
                f"Failed to create ChromaDB client: {str(e)}"
            ) from e

    def _initialize_collection(self, collection_name: str) -> ChromaCollection:
        """
        Initialize or connect to a ChromaDB collection.

        Args:
            collection_name: Name of the collection to initialize

        Returns:
            ChromaCollection: ChromaDB collection for vector operations

        Raises:
            InitializationError: If collection initialization fails
        """
        try:
            # Create or get collection
            metadata: Dict[str, Any] = {
                "dimension": self.dimension,
                "model": self.model_name,
            }
            collection = self.client.get_or_create_collection(
                collection_name, metadata=metadata
            )
            self.logger.info(
                f"Connected to collection '{collection_name}' with {collection.count()} vectors"
            )
            return collection
        except Exception as e:
            raise InitializationError(
                f"Failed to initialize collection: {str(e)}"
            ) from e

    def _validate_vector_dimension(
        self, vector: NDArray[np.float32], context: str = "Vector"
    ) -> None:
        """
        Validate that vector dimensions match expected dimensions.

        Args:
            vector: The vector to validate
            context: Description of the vector for error messages

        Raises:
            DimensionMismatchError: If vector dimensions are incorrect
        """
        if len(vector.shape) != 1 or vector.shape[0] != self.dimension:
            raise DimensionMismatchError(
                f"{context} dimension {vector.shape[0]} doesn't match expected {self.dimension}"
            )

    def format_with_instruction(
        self, text: str, template_key: str = "search", is_query: bool = True
    ) -> str:
        """
        Format text with instruction templates for embedding models.

        Applies the appropriate instruction template to optimize the text
        for the specific embedding model being used. This is especially
        important for instruction-tuned models that expect specific formats.

        Args:
            text: Text to format
            template_key: Template type to use (search, definition, etc.)
            is_query: Whether this is a query (vs. document)

        Returns:
            Formatted text ready for embedding
        """
        # Don't format if no templates available
        if not self.instruction_templates:
            return text
        # Get template or use default
        # Get template or use default
        template: Optional[InstructionTemplate] = self.instruction_templates.get(
            template_key, self.instruction_templates.get("default", None)
        )

        # Return unformatted text if template is None
        if template is None:
            return text

        # Format using appropriate template parts
        task = template.get("task", "")
        prefix = template.get("query_prefix" if is_query else "document_prefix", "")

        if task and prefix:
            return f"{task}\n{prefix}{text}"
        elif prefix:
            return f"{prefix}{text}"
        else:
            return text

    def embed_text(
        self,
        text: str,
        template_key: str = "search",
        is_query: bool = True,
        normalize: bool = True,
    ) -> NDArray[np.float32]:
        """
        Transforms raw text into a high-dimensional vector representation

        Applies the appropriate instruction template based on the task type,
        then generates a vector embedding using the configured model.

        Args:
            text: Text to embed
            template_key: Template type to use (search, definition, etc.)
            is_query: Whether this is a query (True) or document (False)
            normalize: Whether to normalize the vector to unit length

        Returns:
            Embedding vector as numpy array (normalized by default)

        Raises:
            ContentProcessingError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ContentProcessingError("Cannot embed empty text")

        try:
            # Format with instruction template if available
            formatted_text = self.format_with_instruction(text, template_key, is_query)

            # Generate embedding
            vector = self.model.encode(  # type: ignore
                formatted_text,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            # Ensure correct type
            vector = vector.astype(np.float32)

            # Validate dimensions
            self._validate_vector_dimension(
                vector,
                context=f"Embedding for '{text[:30]}{'...' if len(text) > 30 else ''}'",
            )

            return vector

        except Exception as e:
            raise ContentProcessingError(f"Failed to embed text: {str(e)}") from e

    def _get_content_info(
        self, content_id: int, content_type: ContentType
    ) -> Dict[str, Any]:
        """
        Retrieve additional information about content for metadata.

        Args:
            content_id: ID of the content item
            content_type: Type of content (word, definition, etc.)

        Returns:
            Dict containing metadata about the content

        Raises:
            ContentProcessingError: If content retrieval fails
        """
        # Skip if no database manager or no connection
        if self.db_manager is None or self.db_manager.connection is None:
            return {}

        try:
            if content_type == "word":
                # Get the word term from the ID
                cursor = self.db_manager.connection.execute(
                    SQL_GET_TERM_BY_ID, (content_id,)
                )
                row = cursor.fetchone()
                if row:
                    return {"term": row[0]}

            elif content_type == "message":
                # Get message text and other metadata
                cursor = self.db_manager.connection.execute(
                    SQL_GET_MESSAGE_TEXT, (content_id,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "text": row[0],
                        "conversation_id": row[1] if len(row) > 1 else None,
                        "speaker": row[2] if len(row) > 2 else None,
                    }

            return {}

        except Exception as e:
            self.logger.warning(
                f"Failed to get content info for {content_type} ID {content_id}: {e}"
            )
            return {}

    def _get_emotion_info(self, item_id: int) -> Dict[str, Any]:
        """
        Get emotional attributes for an item if available.

        Args:
            item_id: ID of the item to get emotions for

        Returns:
            Dict containing emotion data (valence, arousal, etc.)
        """
        # Skip if no emotion manager
        if self.emotion_manager is None:
            return {}

        try:
            # Get emotion data from the emotion manager
            emotion_data = self.emotion_manager.get_word_emotion(item_id)
            if emotion_data:
                return {
                    "emotion_valence": emotion_data.get("valence"),
                    "emotion_arousal": emotion_data.get("arousal"),
                    "emotion_label": emotion_data.get("label"),
                }
            return {}

        except Exception as e:
            self.logger.warning(f"Failed to get emotion info for ID {item_id}: {e}")
            return {}

    def process_word_entry(self, entry: WordEntryDict) -> List[VectorItem]:
        """
        Process a word entry into vector items for storage.

        Creates separate vector items for the word, its definition, and usage examples,
        each with appropriate metadata for filtering and retrieval.

        Args:
            entry: Word entry dictionary with all word data

        Returns:
            List of items ready for vector storage, each containing vector, metadata, and text

        Raises:
            ContentProcessingError: If processing fails
        """
        word_id = entry["id"]
        term = entry["term"]
        definition = entry["definition"]

        # Handle usage_examples that could be either string or list
        usage_examples = entry.get("usage_examples", "")
        usage_examples = "; ".join(usage_examples)
        examples = self._parse_usage_examples(usage_examples)

        language = entry.get("language", "en")

        # Items to be vectorized
        vector_items: List[VectorItem] = []

        try:
            # 1. Process the word term itself
            word_embedding = self.embed_text(term, template_key="term", normalize=True)

            # Combine standard metadata with emotion data
            emotion_info = self._get_emotion_info(int(word_id))
            word_metadata: VectorMetadata = {
                "original_id": int(word_id),
                "content_type": "word",
                "term": term,
                "language": language,
                "timestamp": time.time(),
                "emotion_valence": emotion_info.get("emotion_valence"),
                "emotion_arousal": emotion_info.get("emotion_arousal"),
                "emotion_label": emotion_info.get("emotion_label"),
            }

            # Add word item
            vector_items.append(
                {
                    "id": f"w_{word_id}",
                    "text": term,
                    "metadata": word_metadata,
                    "vector": word_embedding,
                }
            )

            # 2. Process the definition if available
            if definition and definition.strip():
                def_embedding = self.embed_text(
                    definition,
                    template_key="definition",
                    normalize=True,
                    is_query=False,
                )

                def_metadata: VectorMetadata = {
                    "original_id": int(word_id),
                    "content_type": "definition",
                    "term": term,
                    "definition": definition,
                    "language": language,
                    "timestamp": time.time(),
                }

                # Add definition item
                vector_items.append(
                    {
                        "id": f"d_{word_id}",
                        "text": definition,
                        "metadata": def_metadata,
                        "vector": def_embedding,
                    }
                )

            # 3. Process each usage example
            for i, example in enumerate(examples):
                if not example.strip():
                    continue

                example_embedding = self.embed_text(
                    example, template_key="example", normalize=True, is_query=False
                )

                example_metadata: VectorMetadata = {
                    "original_id": int(word_id),
                    "content_type": "example",
                    "term": term,
                    "language": language,
                    "timestamp": time.time(),
                }

                # Add example item
                vector_items.append(
                    {
                        "id": f"e_{word_id}_{i}",
                        "text": example,
                        "metadata": example_metadata,
                        "vector": example_embedding,
                    }
                )

            return vector_items

        except Exception as e:
            raise ContentProcessingError(
                f"Failed to process word entry for '{term}': {str(e)}"
            ) from e

    def _parse_usage_examples(self, examples_string: str) -> List[str]:
        """
        Parse usage examples from a semicolon-separated string.

        Args:
            examples_string: String containing semicolon-separated examples

        Returns:
            List of individual usage examples
        """
        if not examples_string:
            return []

        # Split by semicolon and strip whitespace
        examples = [ex.strip() for ex in examples_string.split(";")]

        # Filter out empty examples
        return [ex for ex in examples if ex]

    def store_word(self, entry: WordEntryDict) -> int:
        """
        Creates vector embeddings for the word term, definition, and usage examples,
        then stores them in the vector database with appropriate metadata.

        This is the main entry point for adding word data to the vector store.

        Args:
            entry: Complete word entry with all data

        Returns:
            int: Number of vectors successfully stored

        Raises:
            ContentProcessingError: If processing or storage fails
        """
        try:
            # Process the word entry into vector items
            vector_items = self.process_word_entry(entry)

            # Store each vector item
            for item in vector_items:
                # Extract vector_id from the id string (e.g., "w_123" -> 123)
                vec_id = item["id"]

                # Convert VectorMetadata to Dict[str, Any]
                metadata_dict = dict(item["metadata"])

                # Call upsert with extracted ID and other components
                self.upsert(
                    vec_id=vec_id,
                    embedding=item["vector"],
                    metadata=metadata_dict,
                    text=item["text"],
                )

            # Return the number of vectors stored
            return len(vector_items)

        except Exception as e:
            raise ContentProcessingError(
                f"Failed to store word '{entry.get('term', '')}': {str(e)}"
            ) from e

    def delete_vectors_for_word(self, word_id: WordID) -> int:
        """
        Delete all vectors associated with a specific word.

        Removes vectors associated with the word from the vector store,
        including all vectors for the term, definition, and examples.

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
            # Convert to string for compatibility with ChromaDB
            word_id_str = str(word_id)
            deleted_count = 0

            # Try to delete vectors by ID pattern
            try:
                # Use metadata filter to find all vectors for this word
                word_filter = {"original_id": int(word_id)}

                # Try to delete vectors by ID and filter
                self.collection.delete(where=word_filter)

                # Count the number of vectors deleted
                deleted_count = 3  # Approximate for term, definition, examples
                self.logger.info(f"Deleted vectors for word {word_id} using filter")

            except Exception as inner_e:
                self.logger.warning(
                    f"Failed to delete vectors by filter for {word_id}: {inner_e}"
                )

                # Fallback: Try explicit ID patterns
                ids_to_delete = [
                    f"w_{word_id_str}",  # Word term
                    f"d_{word_id_str}",  # Definition
                ]

                # Add potential example IDs
                for i in range(10):  # Assume max 10 examples per word
                    ids_to_delete.append(f"e_{word_id_str}_{i}")

                try:
                    self.collection.delete(ids=ids_to_delete)
                    deleted_count = len(ids_to_delete)
                    self.logger.info(
                        f"Deleted vectors for word {word_id} using explicit IDs"
                    )
                except Exception as id_e:
                    self.logger.error(
                        f"Failed to delete vectors by ID for {word_id}: {id_e}"
                    )
                    raise VectorStoreError(
                        "Neither ID nor filter-based deletion succeeded"
                    )

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
        ChromaDB compatibility, and handles the low-level storage operations.

        Args:
            vec_id: Unique identifier for the vector
            embedding: Vector embedding to store
            metadata: Optional metadata to associate with the vector
            text: Optional text content to associate with the vector

        Raises:
            UpsertError: If ChromaDB operation fails
        """
        # Validate vector dimensions
        self._validate_vector_dimension(embedding, "Embedding")

        # Convert ID to string for ChromaDB
        vec_id_str = str(vec_id)

        # Prepare metadata - ensure all values are compatible with ChromaDB
        sanitized_metadata = self._sanitize_metadata(metadata or {})

        try:
            # Upsert into collection
            self.collection.upsert(
                ids=[vec_id_str],
                embeddings=[embedding.tolist()],
                metadatas=[sanitized_metadata] if sanitized_metadata else None,
                documents=[text] if text else None,
            )

            # Persist to disk if using persistent storage
            self._persist_if_needed()

        except Exception as e:
            raise UpsertError(f"Failed to store vector: {str(e)}") from e

    def _sanitize_metadata(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Union[str, int, float, bool]]:
        """
        Sanitize metadata to ensure compatibility with ChromaDB.

        Args:
            metadata: Raw metadata dictionary

        Returns:
            Dictionary with ChromaDB-compatible values
        """
        result: Dict[str, Union[str, int, float, bool]] = {}

        for key, value in metadata.items():
            # Skip None values
            if value is None:
                continue

            # Convert values to compatible types
            if isinstance(value, (str, int, float, bool)):
                result[key] = value
            else:
                # Convert other types to string
                result[key] = str(value)

        return result

    # Type overloads for search method to enable different call patterns
    @overload
    def search(
        self,
        *,
        query_text: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResultDict]: ...

    @overload
    def search(
        self,
        query_vector: NDArray[np.float32],
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
        that are semantically similar to the query. Supports two search modes:

        1. Direct vector search - when you already have an embedding
        2. Text search - converts text to embedding then searches

        Args:
            query_vector: Vector representation to search against
            query_text: Text to search for (will be converted to a vector)
            k: Number of results to return
            filter_metadata: Optional metadata filters for narrowing results

        Returns:
            List of search results with distance scores and metadata

        Raises:
            SearchError: If neither query_vector nor query_text is provided
            DimensionMismatchError: If query vector dimensions are incorrect

        Examples:
            >>> # Search by text
            >>> results = vector_store.search(
            ...     query_text="machine learning algorithm",
            ...     k=5,
            ...     filter_metadata={"content_type": "definition"}
            ... )

            >>> # Search by vector
            >>> results = vector_store.search(query_vector=embedding)
        """
        if query_vector is None and query_text is None:
            raise SearchError(
                "Search requires either query_vector or query_text parameter"
            )

        if query_vector is not None and query_text is not None:
            raise SearchError("Provide either query_vector or query_text, not both")

        # Prepare search vector
        search_vector = self._prepare_search_vector(query_vector, query_text)

        # Execute the search
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
            SearchError: If neither input is provided
            DimensionMismatchError: If vector has incorrect dimensions
        """
        # Case 1: Direct vector provided
        if query_vector is not None:
            self._validate_vector_dimension(query_vector, context="Query vector")
            return query_vector

        # Case 2: Text provided - convert to vector
        assert query_text is not None, "Both query_vector and query_text cannot be None"

        try:
            embedded_vector = self.embed_text(
                query_text, template_key="search", is_query=True, normalize=True
            )
            self._validate_vector_dimension(embedded_vector, context="Embedded query")
            return embedded_vector

        except Exception as e:
            raise SearchError(f"Failed to prepare search vector: {str(e)}") from e

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
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of search results with distance scores and metadata

        Raises:
            SearchError: If ChromaDB operation fails
        """
        try:
            # Enforce reasonable upper limit on results
            max_allowed = getattr(config.vectorizer, "max_results", 100)
            adjusted_k = min(k, max_allowed)

            # Adjust filter format for ChromaDB
            where_clause: Optional[Dict[str, Any]] = None
            if filter_metadata:
                where_clause = {}
                for key, value in filter_metadata.items():
                    if value is not None:
                        where_clause[key] = value

            # Execute search (measure time for logging)
            start_time = time.time()

            query_results = self.collection.query(
                query_embeddings=[search_vector.tolist()],
                n_results=adjusted_k,
                where=where_clause,
            )

            search_time = time.time() - start_time

            # Process results into a consistent format
            results = self._process_chromadb_results(query_results)

            # Log search statistics
            self.logger.debug(
                f"Vector search took {search_time:.2f}s for k={adjusted_k}, "
                f"found {len(results)} results"
            )

            return results

        except Exception as e:
            error_msg = f"Vector search failed: {str(e)}"
            self.logger.error(
                f"{error_msg} | Vector shape: {search_vector.shape} | "
                f"Filter: {filter_metadata}"
            )
            raise SearchError(error_msg) from e

    def _process_chromadb_results(
        self, query_results: Dict[str, List[Any]]
    ) -> List[SearchResultDict]:
        """
        Process raw ChromaDB query results into a standardized format.

        Args:
            query_results: Raw results from ChromaDB query

        Returns:
            List of standardized search result dictionaries
        """
        results: List[SearchResultDict] = []

        # Process results if available
        if not query_results or "ids" not in query_results:
            return results

        # Extract result components
        ids = query_results.get("ids", [[]])[0]
        distances = query_results.get("distances", [[]])[0]
        metadatas = query_results.get("metadatas", [[None] * len(ids)])[0]
        documents = query_results.get("documents", [[None] * len(ids)])[0]

        # Convert similarities to distances if needed
        if distances and min(distances) >= 0 and max(distances) <= 1:
            distances = self._convert_similarities_to_distances(distances)

        # Build result list
        for i, result_id in enumerate(ids):
            try:
                # Extract the original numeric ID if possible
                original_id = (
                    int(result_id.split("_")[1]) if "_" in result_id else int(result_id)
                )

                # Get metadata and text
                metadata = (
                    cast(Optional[VectorMetadata], metadatas[i])
                    if i < len(metadatas)
                    else None
                )
                document = documents[i] if i < len(documents) else None

                # Calculate distance (handle any missing values)
                distance = distances[i] if i < len(distances) else 1.0

                # Add to results
                results.append(
                    SearchResultDict(
                        id=original_id,
                        distance=distance,
                        metadata=metadata,
                        text=document,
                    )
                )
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Error processing search result {result_id}: {e}")

        return results

    def get_legacy_search_results(
        self, query_vector: NDArray[np.float32], k: int = 5
    ) -> SearchResult:
        """
        Legacy method for backward compatibility with older code.

        Returns search results in the original (id, distance) tuple format.

        Args:
            query_vector: Query embedding to search against
            k: Number of results to return

        Returns:
            List of (id, distance) tuples

        Raises:
            SearchError: If search fails
        """
        results = self.search(query_vector=query_vector, k=k)

        # Convert to legacy format
        legacy_results: SearchResult = [
            (result["id"], result["distance"]) for result in results
        ]

        return legacy_results

    def _convert_similarities_to_distances(
        self, similarities: List[float]
    ) -> List[float]:
        """
        Convert similarity scores (0-1, higher is better) to distances (lower is better).

        Args:
            similarities: List of similarity scores

        Returns:
            List of equivalent distance scores
        """
        return [1.0 - sim for sim in similarities]

    def _persist_if_needed(self) -> None:
        """
        Persist vector store to disk when appropriate.

        Only applies to persistent storage types.
        """
        if self.storage_type != StorageType.MEMORY and hasattr(self.client, "persist"):
            try:
                self.client.persist()
            except Exception as e:
                self.logger.warning(
                    f"Failed to persist vector store to {self.index_path}: {str(e)}"
                )

    def __del__(self) -> None:
        """
        Ensures vector data is persisted before the object is destroyed

        Called automatically when the object is garbage collected.
        """
        try:
            self._persist_if_needed()
        except Exception:
            # Ignore errors during cleanup
            pass

    @staticmethod
    def is_valid_vector_store(obj: Any) -> bool:
        """
        Check if an object is a valid VectorStore instance.

        Args:
            obj: Object to check

        Returns:
            bool: True if the object is a valid VectorStore, False otherwise
        """
        if not isinstance(obj, VectorStore):
            return False

        # Additional checks for required attributes and methods
        required_attrs = ["upsert", "search", "delete_vectors_for_word"]
        return all(hasattr(obj, attr) for attr in required_attrs)
