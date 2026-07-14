"""
Vector Worker module for Word Forge.

This module provides thread-based workers for continuous processing of word data
into vector embeddings to enable semantic search capabilities. It handles the
connection between the database layer and the vector store, managing the lifecycle
of embedding generation and storage.

Architecture:
    ┌─────────────────────┐
    │    VectorWorker     │
    └──────────┬──────────┘
               │
    ┌──────────┴──────────┐
    │      Components     │
    └─────────────────────┘
    ┌───────┬────────┬───────┐
    │  DB   │Embedder│Vector │
    │Manager│        │Store  │
    └───────┴────────┴───────┘
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Protocol, TypedDict, Union, cast, final

import numpy as np
from numpy.typing import NDArray

from word_forge.database.database_manager import DBManager
from word_forge.vectorizer.embedding_models import (
    DEFAULT_EMBEDDING_MODEL,
    format_embedding_text,
)
from word_forge.vectorizer.model_utils import get_embedding_dimension
from word_forge.vectorizer.vector_store import VectorStore


class VectorState(Enum):
    """Worker lifecycle states for monitoring and control."""

    RUNNING = auto()
    STOPPED = auto()
    ERROR = auto()

    def __str__(self) -> str:
        """Return lowercase state name for consistent string representation."""
        return self.name.lower()


class EmbeddingError(Exception):
    """
    Raised when embedding generation fails.

    This exception captures issues with the embedding process, such as model errors,
    empty text, or other failures in generating vector representations.
    """

    pass


class VectorStoreError(Exception):
    """
    Raised when vector storage operations fail.

    This exception is used when operations on the vector store like insertion,
    retrieval, or search cannot be completed successfully.
    """

    pass


class DatabaseError(Exception):
    """
    Raised when database operations fail.

    This exception indicates issues with the underlying database operations,
    such as connection failures, query errors, or data inconsistencies.
    """

    pass


class WordData(Protocol):
    """
    Protocol defining the required structure for words.

    This protocol establishes the contract that any word-like object must
    provide certain properties for compatibility with the vector worker.
    """

    @property
    def id(self) -> int:
        """Return the unique identifier for the word."""
        ...

    @property
    def term(self) -> str:
        """Return the word term."""
        ...

    @property
    def definition(self) -> str:
        """Return the word definition."""
        ...

    @property
    def usage_examples(self) -> List[str]:
        """Return usage examples for the word."""
        ...


@dataclass(frozen=True)
class Word:
    """
    Data object for storing word information.

    A simple, immutable container for word-related data including the term,
    definition, and examples of usage.

    Attributes:
        id: Unique identifier for the word
        term: The word or phrase itself
        definition: The meaning or explanation of the word
        usage_examples: List of example sentences using the word
    """

    id: int
    term: str
    definition: str
    usage_examples: List[str]


class ProcessingResult(Enum):
    """
    Possible outcomes when processing a word.

    These enum values represent the different states that can result
    from attempting to process a word entry.
    """

    SUCCESS = auto()  # Word was successfully processed and stored
    EMBEDDING_ERROR = auto()  # Failed to generate embedding
    STORAGE_ERROR = auto()  # Failed to store embedding


class VectorWorkerStatus(TypedDict):
    """
    Type definition for worker status information.

    This dictionary type defines the structure of status information
    returned by the vector worker.
    """

    running: bool  # Whether the worker is currently active
    processed_count: int  # Number of words processed in current cycle
    successful_count: int  # Number of successful operations
    error_count: int  # Number of failed operations
    last_update: Optional[float]  # Timestamp of last update
    uptime: Optional[float]  # Seconds since worker was started
    state: str  # Current worker state (running, stopped, error)


@dataclass
class ProcessingStats:
    """
    Statistics about word processing operations.

    Tracks metrics about processing performance including success and failure counts
    and categorized error information.

    Attributes:
        processed: Total number of words processed
        successful: Number of successfully processed words
        failed: Number of failed processing attempts
        errors: Dictionary mapping error types to counts
        last_update: Timestamp of the last update
    """

    processed: int = 0
    successful: int = 0
    failed: int = 0
    errors: Dict[str, int] = field(default_factory=dict)
    last_update: Optional[float] = None

    def record_result(self, word_id: int, result: ProcessingResult) -> None:
        """
        Record the result of processing a word.

        Updates the statistics counters based on the processing outcome.

        Args:
            word_id: Identifier of the processed word
            result: Outcome of the processing operation
        """
        self.processed += 1
        self.last_update = time.time()

        if result == ProcessingResult.SUCCESS:
            self.successful += 1
        else:
            self.failed += 1
            error_type = result.name
            self.errors[error_type] = self.errors.get(error_type, 0) + 1

    def clear(self) -> None:
        """
        Reset all statistics.

        Sets all counters back to zero and clears the error tracking.
        """
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.errors.clear()
        self.last_update = None


class WordRow(TypedDict):
    """
    Structure of a word row from the database.

    This type defines the expected structure of dictionary data
    returned from database queries about words.
    """

    id: int  # Unique identifier
    term: str  # The word or phrase
    definition: str  # Definition text
    usage_examples: str  # Semicolon-separated examples


class Embedder(Protocol):
    """
    Protocol for text embedding generators.

    This protocol defines the interface required for any component that
    can generate vector embeddings from text.
    """

    def embed(self, text: str) -> NDArray[np.float32]:
        """
        Generate an embedding vector for the given text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as numpy array

        Raises:
            EmbeddingError: If embedding generation fails
        """
        ...


class VectorWorkerInterface(Protocol):
    """
    Protocol defining the required interface for vector workers.

    This protocol establishes the minimal interface that must be implemented
    by any vector worker component.
    """

    def start(self) -> None:
        """Start the worker thread."""
        ...

    def stop(self) -> None:
        """Signal the worker to stop processing."""
        ...

    def get_status(self) -> VectorWorkerStatus:
        """Return current worker status information."""
        ...

    def is_alive(self) -> bool:
        """Check if the worker thread is running."""
        ...


@dataclass(frozen=True)
class _VectorStoreEmbedder:
    """Adapt the store-owned model to the worker's legacy embedder protocol."""

    vector_store: VectorStore

    @property
    def dimension(self) -> int:
        """Return the store model's authoritative embedding dimension."""

        return self.vector_store.dimension

    @property
    def model_name(self) -> str:
        """Return the store-owned model identifier."""

        return self.vector_store.model_name

    def embed(self, text: str) -> NDArray[np.float32]:
        """Embed a lexical document through the store's single model instance."""

        return self.vector_store.embed_text(
            text,
            template_key="definition",
            is_query=False,
        )


@final
class VectorWorker(threading.Thread):
    """
    Continuously scans the DB for words, generates embeddings, and stores them.

    This worker runs as a daemon thread that polls a database at regular intervals,
    generates vector embeddings for each word (combining term, definition, and usage examples),
    and stores these embeddings in a vector store for similarity search.

    Attributes:
        db: Database manager providing access to word data
        vector_store: Vector store for saving embeddings
        embedder: Text embedding generator
        poll_interval: Time in seconds between database polling cycles
        logger: Logger for error reporting and status updates
        stats: Statistics about processing operations
    """

    def __init__(
        self,
        db: DBManager,
        vector_store: VectorStore,
        embedder: Optional[Union[Embedder, str]] = None,
        poll_interval: Optional[float] = None,
        daemon: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the vector worker.

        Args:
            db: Database manager providing access to word data
            vector_store: Vector store for saving embeddings
            embedder: Optional custom generator or model identifier. A matching
                model identifier reuses the model already owned by ``vector_store``.
            poll_interval: Time in seconds between database polling cycles (defaults to 10.0)
            daemon: Whether to run as a daemon thread
            logger: Optional logger for error reporting
        """
        super().__init__(daemon=daemon)
        self.db = db
        self.vector_store = vector_store

        if embedder is None or isinstance(embedder, str):
            requested_model = embedder or vector_store.model_name
            if requested_model.strip().casefold() != vector_store.model_name.casefold():
                raise EmbeddingError(
                    f"Worker model '{requested_model}' does not match VectorStore model "
                    f"'{vector_store.model_name}'. Construct one VectorStore with the "
                    "requested model so indexing uses a single embedding space."
                )
            self.embedder: Embedder = _VectorStoreEmbedder(vector_store)
        else:
            self.embedder = embedder
            embedder_dimension = getattr(embedder, "dimension", None)
            if (
                isinstance(embedder_dimension, int)
                and embedder_dimension != vector_store.dimension
            ):
                raise EmbeddingError(
                    f"Custom embedder dimension {embedder_dimension} does not match "
                    f"VectorStore dimension {vector_store.dimension}."
                )
            custom_model = getattr(embedder, "model_name", None)
            if (
                isinstance(custom_model, str)
                and custom_model.strip().casefold()
                != vector_store.model_name.casefold()
            ):
                raise EmbeddingError(
                    f"Custom embedder model '{custom_model}' does not match "
                    f"VectorStore model '{vector_store.model_name}'."
                )
            if custom_model is None and not vector_store.demo_mode:
                raise EmbeddingError(
                    "Custom embedders without a model identity are limited to "
                    "explicit demo-mode stores."
                )

        self.poll_interval = poll_interval or 10.0

        self._stop_flag = False
        self._current_state = VectorState.STOPPED
        self._status_lock = threading.RLock()
        self._start_time: Optional[float] = None
        self.last_processed: Optional[float] = None

        self.logger = logger or logging.getLogger(__name__)
        self.stats = ProcessingStats()

    def run(self) -> None:
        """
        Run the worker thread until stopped.

        Continuously fetches words from the database, generates embeddings,
        and stores them in the vector store with each word's ID.
        """
        with self._status_lock:
            self._start_time = time.time()
            self._current_state = VectorState.RUNNING

        self.logger.info("Vector worker started")
        self._warn_about_demo_configuration()

        while not self._stop_flag:
            try:
                # Clear statistics for this cycle
                self.stats.clear()

                # Process all words
                words = self._get_all_words()
                self._process_words(words)

                # Log summary of this processing cycle
                self._log_cycle_summary()

                # Record completion time
                self.last_processed = time.time()

                # Wait before next cycle
                time.sleep(self.poll_interval)
            except Exception as e:
                with self._status_lock:
                    self._current_state = VectorState.ERROR

                self.logger.error(f"Error in vector worker cycle: {str(e)}")
                # Continue running despite errors
                time.sleep(max(1.0, self.poll_interval / 2))

        with self._status_lock:
            self._current_state = VectorState.STOPPED

        self.logger.info("Vector worker stopped")

    def stop(self) -> None:
        """Signal the worker thread to stop after the current cycle."""
        self._stop_flag = True
        self.logger.info("Vector worker stop requested")

    def _warn_about_demo_configuration(self) -> None:
        """Log warnings when the worker is running in demo or non-deterministic mode."""

        if isinstance(self.embedder, SimpleEmbedder):
            self.logger.warning(
                "VectorWorker is using %s; its vectors are not reproducible.",
                type(self.embedder).__name__,
            )

        if getattr(self.vector_store, "demo_mode", False):
            self.logger.warning(
                "VectorWorker is running against a demo-mode in-memory VectorStore; vectors will not persist."
            )

    def get_status(self) -> VectorWorkerStatus:
        """
        Return the current status of the vector worker.

        Returns:
            Dictionary containing operational metrics including:
            - running: Whether the worker is active
            - processed_count: Number of words processed
            - successful_count: Number of successful embeddings
            - error_count: Number of encountered errors
            - last_update: Timestamp of last successful update
            - uptime: Seconds since thread start if running
            - state: Current worker state as string
        """
        with self._status_lock:
            uptime = None
            if self._start_time:
                uptime = time.time() - self._start_time

            status: VectorWorkerStatus = {
                "running": self.is_alive() and not self._stop_flag,
                "processed_count": self.stats.processed,
                "successful_count": self.stats.successful,
                "error_count": self.stats.failed,
                "last_update": self.stats.last_update,
                "uptime": uptime,
                "state": str(self._current_state),
            }

            return status

    def _get_all_words(self) -> List[Word]:
        """
        Fetch new or updated words from the database.

        Returns:
            List of Word objects containing term, definition, and usage examples

        Raises:
            DatabaseError: If database operations fail
        """
        try:
            if self.last_processed is None:
                rows = self.db.get_all_words()
            else:
                rows = self.db.get_updated_words(self.last_processed)
            return [self._convert_row_to_word(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to fetch words from database: {str(e)}")
            raise DatabaseError(f"Database fetch failed: {str(e)}") from e

    def _convert_row_to_word(self, row: WordRow) -> Word:
        """
        Convert a database row to a Word object.

        Args:
            row: Database row containing word data

        Returns:
            Word object with parsed data
        """
        return Word(
            id=row["id"],
            term=row["term"],
            definition=row["definition"],
            usage_examples=self._parse_usage_examples(row["usage_examples"]),
        )

    def _parse_usage_examples(self, examples_string: str) -> List[str]:
        """
        Parse usage examples from a semicolon-separated string.

        Args:
            examples_string: String containing examples separated by semicolons

        Returns:
            List of individual usage examples
        """
        return examples_string.split("; ") if examples_string else []

    def _process_words(self, words: List[Word]) -> None:
        """
        Process a batch of words into vector embeddings.

        Generates embeddings for each word and stores them in the vector store.
        The normal path delegates to the model already owned by ``VectorStore``
        so indexing never allocates a duplicate transformer or mixes spaces.

        Args:
            words: List of words to process
        """
        for word in words:
            # Prepare embedding text
            text = self._prepare_embedding_text(word)

            try:
                vector = self.embedder.embed(text)

                # Store embedding with metadata for better retrieval
                metadata = {
                    "content_type": "word",
                    "term": word.term,
                    "original_id": word.id,
                }
                self.vector_store.upsert(word.id, vector, metadata=metadata, text=text)
                self.stats.record_result(word.id, ProcessingResult.SUCCESS)

            except EmbeddingError as e:
                self.logger.error(
                    f"Embedding failed for word {word.term} (ID: {word.id}): {str(e)}"
                )
                self.stats.record_result(word.id, ProcessingResult.EMBEDDING_ERROR)
            except Exception as e:
                error_msg = f"Vector storage failed for word {word.term} (ID: {word.id}): {str(e)}"
                if "object has no attribute 'upsert'" in str(e):
                    error_msg += f" (vector_store has type {type(self.vector_store).__name__}, should be VectorStore)"
                elif "dimension" in str(e).lower():
                    # Provide helpful guidance for dimension mismatches
                    embedder_dim = getattr(self.embedder, "dimension", "unknown")
                    store_dim = getattr(self.vector_store, "dimension", "unknown")
                    error_msg += (
                        f" (embedder dimension: {embedder_dim}, "
                        f"store dimension: {store_dim})"
                    )
                self.logger.error(error_msg)
                self.stats.record_result(word.id, ProcessingResult.STORAGE_ERROR)

    def _prepare_embedding_text(self, word: Word) -> str:
        """
        Prepare the text that will be embedded.

        Args:
            word: Word data to prepare for embedding

        Returns:
            Combined text string for embedding
        """
        return f"{word.term} {word.definition} {' '.join(word.usage_examples)}"

    def _log_cycle_summary(self) -> None:
        """Log a summary of the current processing cycle."""
        if self.stats.processed == 0:
            self.logger.info("No words processed in this cycle")
            return

        success_rate = (self.stats.successful / self.stats.processed) * 100
        self.logger.info(
            f"Cycle complete: {self.stats.processed} words processed, "
            f"{self.stats.successful} successful ({success_rate:.1f}%), "
            f"{self.stats.failed} failed"
        )

        if self.stats.errors:
            error_details = ", ".join(
                f"{count} {error}" for error, count in self.stats.errors.items()
            )
            self.logger.info(f"Errors: {error_details}")


class SimpleEmbedder:
    """
    Simple embedding generator for demonstration purposes.

    Creates random vectors of specified dimension. In a production environment,
    this would be replaced with a proper embedding model.

    Attributes:
        dimension: Size of the embedding vector
    """

    def __init__(self, dimension: int = 768):
        """
        Initialize with specified embedding dimension.

        Args:
            dimension: Size of the embedding vector
        """
        self.dimension = dimension

    def embed(self, text: str) -> NDArray[np.float32]:
        """
        Generate a random embedding vector for the given text.

        Args:
            text: The text to embed

        Returns:
            Random embedding vector as numpy array

        Raises:
            EmbeddingError: If text is empty or embedding fails
        """
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        try:
            # For demonstration purposes, create a random vector
            return np.random.rand(self.dimension).astype(np.float32)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}") from e


class TransformerEmbedder:
    """
    High-quality embedding generator using transformer models.

    Uses sentence-transformer models for sophisticated text embeddings
    optimized for search and retrieval tasks.

    Attributes:
        model: Sentence transformer model instance
        dimension: Dimension of the generated embeddings
        model_name: Name of the model being used
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initialize with a pretrained transformer model.

        Args:
            model_name: Name of the pretrained model to use. Should be a valid
                HuggingFace model path (e.g., 'sentence-transformers/all-MiniLM-L6-v2'
                or 'intfloat/multilingual-e5-large-instruct').

        Raises:
            EmbeddingError: If model initialization fails with details about the cause
        """
        self.model_name = model_name

        # Try to import sentence_transformers
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingError(
                "Missing required package: sentence-transformers.\n"
                "To fix this, install the vector dependencies:\n"
                '    pip install "word_forge[vector]"\n'
                "Or install sentence-transformers directly:\n"
                "    pip install sentence-transformers"
            ) from e

        # Try to load the model with detailed error handling
        # Note: We use string matching to detect error types because the
        # sentence_transformers library raises generic OSError exceptions.
        # This approach provides helpful user guidance while still falling
        # back to the original error message if patterns don't match.
        try:
            self.model = SentenceTransformer(model_name)  # type: ignore
        except OSError as e:
            error_msg = str(e)
            # Check for common error patterns in HuggingFace error messages
            if (
                "not a local folder" in error_msg
                or "not a valid model identifier" in error_msg
            ):
                raise EmbeddingError(
                    f"Model not found: '{model_name}'.\n"
                    "This model name is not recognized by HuggingFace.\n\n"
                    "Common fixes:\n"
                    "  1. Use the full model path (e.g., 'sentence-transformers/all-MiniLM-L6-v2')\n"
                    "  2. Check spelling and capitalization\n"
                    "  3. Verify the model exists at https://huggingface.co/models\n\n"
                    "Popular embedding models:\n"
                    "  - intfloat/multilingual-e5-small (default multilingual, 384 dim)\n"
                    "  - sentence-transformers/all-MiniLM-L6-v2 (fast, 384 dim)\n"
                    "  - sentence-transformers/all-mpnet-base-v2 (balanced, 768 dim)\n"
                    "  - intfloat/multilingual-e5-large-instruct (multilingual, 1024 dim)"
                ) from e
            elif "private" in error_msg.lower() or "gated" in error_msg.lower():
                raise EmbeddingError(
                    f"Access denied for model: '{model_name}'.\n"
                    "This model may be private or require authentication.\n\n"
                    "To fix this:\n"
                    "  1. Log in with: huggingface-cli login\n"
                    "  2. Or use a token: export HUGGING_FACE_HUB_TOKEN=your_token\n"
                    "  3. Or choose a public model instead"
                ) from e
            else:
                raise EmbeddingError(
                    f"Failed to load model '{model_name}': {error_msg}"
                ) from e
        except ConnectionError as e:
            raise EmbeddingError(
                f"Network error while loading model '{model_name}'.\n"
                "Could not connect to HuggingFace Hub.\n\n"
                "To fix this:\n"
                "  1. Check your internet connection\n"
                "  2. Try again later if HuggingFace is experiencing issues\n"
                "  3. If behind a proxy, configure HTTPS_PROXY environment variable"
            ) from e
        except Exception as e:
            raise EmbeddingError(
                f"Failed to initialize embedding model '{model_name}': {str(e)}"
            ) from e

        # Get the actual dimension from the loaded model
        try:
            model_dim = get_embedding_dimension(self.model)
            if model_dim is None:
                raise EmbeddingError(
                    f"Model '{model_name}' returned invalid dimension: {model_dim}"
                )
            self.dimension: int = model_dim
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Could not determine embedding dimension for model '{model_name}': {e}"
            ) from e

    def embed(self, text: str) -> NDArray[np.float32]:
        """
        Generate a high-quality embedding vector for the given text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as numpy array

        Raises:
            EmbeddingError: If text is empty or embedding fails
        """
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        try:
            formatted_text = format_embedding_text(
                self.model_name,
                text,
                is_query=False,
            )

            # Generate embedding and return as numpy array
            embedding = self.model.encode(  # type: ignore
                sentences=formatted_text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return cast(NDArray[np.float32], embedding)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}") from e
