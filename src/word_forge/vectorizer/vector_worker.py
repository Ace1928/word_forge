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
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Protocol, TypedDict, cast, final

import numpy as np
import torch
from numpy.typing import NDArray

from word_forge.database.db_manager import DBManager
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
        embedder: Embedder,
        poll_interval: Optional[float] = None,
        daemon: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the vector worker.

        Args:
            db: Database manager providing access to word data
            vector_store: Vector store for saving embeddings
            embedder: Text embedding generator
            poll_interval: Time in seconds between database polling cycles (defaults to 10.0)
            daemon: Whether to run as a daemon thread
            logger: Optional logger for error reporting
        """
        super().__init__(daemon=daemon)
        self.db = db
        self.vector_store = vector_store
        self.embedder = embedder
        self.poll_interval = poll_interval or 10.0

        self._stop_flag = False
        self._current_state = VectorState.STOPPED
        self._status_lock = threading.RLock()
        self._start_time: Optional[float] = None

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

        while not self._stop_flag:
            try:
                # Clear statistics for this cycle
                self.stats.clear()

                # Process all words
                words = self._get_all_words()
                self._process_words(words)

                # Log summary of this processing cycle
                self._log_cycle_summary()

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
        Fetch all words from the database.

        Returns:
            List of Word objects containing term, definition, and usage examples

        Raises:
            DatabaseError: If database operations fail
        """
        try:
            rows = self.db.get_all_words()
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
        Generate and store embeddings for all words.

        Args:
            words: List of Word objects to process
        """
        self.logger.info(f"Processing {len(words)} words")

        for word in words:
            try:
                # Combine word data into a single text for embedding
                text = self._prepare_embedding_text(word)

                # Generate embedding vector
                try:
                    vector = self.embedder.embed(text)
                except Exception as e:
                    self.logger.error(
                        f"Embedding failed for word {word.term} (ID: {word.id}): {str(e)}"
                    )
                    self.stats.record_result(word.id, ProcessingResult.EMBEDDING_ERROR)
                    continue

                # Store embedding in vector store
                try:
                    self.vector_store.upsert(word.id, vector)
                    self.stats.record_result(word.id, ProcessingResult.SUCCESS)
                except Exception as e:
                    self.logger.error(
                        f"Vector storage failed for word {word.term} (ID: {word.id}): {str(e)}"
                    )
                    self.stats.record_result(word.id, ProcessingResult.STORAGE_ERROR)

            except Exception as e:
                self.logger.error(
                    f"Unexpected error processing word {word.term} (ID: {word.id}): {str(e)}"
                )
                # Continue with next word despite errors

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

    Uses the Multilingual-E5-large-instruct model for sophisticated,
    multilingual text embeddings optimized for search and retrieval tasks.

    Attributes:
        model: Sentence transformer model instance
        dimension: Dimension of the generated embeddings
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct"):
        """
        Initialize with a pretrained transformer model.

        Args:
            model_name: Name of the pretrained model to use

        Raises:
            EmbeddingError: If model initialization fails
        """
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                model_name  # type: ignore
            )  # Multilingual-E5-large-instruct
            # Multilingual-E5-large-instruct has embedding dimension of 1024
            self.dimension = 1024
        except ImportError:
            raise EmbeddingError(
                "Failed to import sentence_transformers. Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise EmbeddingError(
                f"Failed to initialize embedding model: {str(e)}"
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
            # Format with task instruction for retrieval optimization
            task = (
                "Given a definition and examples, retrieve related terms and concepts"
            )
            formatted_text = f"Instruct: {task}\nQuery: {text}"

            # Generate embedding and return as numpy array
            embedding = self.model.encode(  # type: ignore
                sentences=formatted_text,  # Use the formatted text directly as sentences
                batch_size=10,  # Batch text optimization
                convert_to_numpy=True,
                normalize_embeddings=True,  # Pre-normalize for cosine similarity
                show_progress_bar=True,  # Show Progress
                output_value="sentence_embedding",  # Ensure correct output
                precision="float32",  # Use float32 for consistency
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            return cast(NDArray[np.float32], embedding)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}") from e


@contextmanager
def temporary_database(path: Path) -> Iterator[Path]:
    """
    Create a temporary database for testing.

    Args:
        path: Path where temporary database will be created

    Yields:
        Path to the created database

    Raises:
        sqlite3.Error: If database creation fails
    """
    conn = None
    try:
        # Create the database
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()

        # Create schema
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY,
                term TEXT NOT NULL,
                definition TEXT NOT NULL,
                usage_examples TEXT
            )
            """
        )

        # Add sample data
        sample_words = [
            (
                1,
                "algorithm",
                "A process or set of rules to be followed for calculations or problem-solving",
                "The sorting algorithm efficiently organized the data; Computer scientists developed a new algorithm for image recognition",
            ),
            (
                2,
                "recursion",
                "The process of defining something in terms of itself",
                "The function uses recursion to calculate factorial; Recursion is a powerful technique in programming",
            ),
        ]

        cursor.executemany(
            "INSERT OR REPLACE INTO words VALUES (?, ?, ?, ?)", sample_words
        )
        conn.commit()

        # Yield the path to the caller
        yield path

    except sqlite3.Error as e:
        if path.exists():
            path.unlink()
        raise e
    finally:
        if conn:
            conn.close()
