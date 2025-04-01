from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Protocol, TypedDict, final

import numpy as np

from word_forge.database.db_manager import DBManager
from word_forge.vectorizer.vector_store import StorageType, VectorStore


class VectorState(Enum):
    """Worker lifecycle states for monitoring and control."""

    RUNNING = auto()
    STOPPED = auto()
    ERROR = auto()

    def __str__(self) -> str:
        """Return lowercase state name for consistent string representation."""
        return self.name.lower()


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""

    pass


class VectorStoreError(Exception):
    """Raised when vector storage operations fail."""

    pass


class DatabaseError(Exception):
    """Raised when database operations fail."""

    pass


class WordData(Protocol):
    """Protocol defining the required structure for words."""

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
    """Data object for storing word information."""

    id: int
    term: str
    definition: str
    usage_examples: List[str]


class ProcessingResult(Enum):
    """Possible outcomes when processing a word."""

    SUCCESS = auto()
    EMBEDDING_ERROR = auto()
    STORAGE_ERROR = auto()


class VectorWorkerStatus(TypedDict):
    """Type definition for worker status information."""

    running: bool
    processed_count: int
    successful_count: int
    error_count: int
    last_update: Optional[float]
    uptime: Optional[float]
    state: str


@dataclass
class ProcessingStats:
    """Statistics about word processing operations."""

    processed: int = 0
    successful: int = 0
    failed: int = 0
    errors: Dict[str, int] = field(default_factory=dict)
    last_update: Optional[float] = None

    def record_result(self, word_id: int, result: ProcessingResult) -> None:
        """Record the result of processing a word."""
        self.processed += 1
        self.last_update = time.time()

        if result == ProcessingResult.SUCCESS:
            self.successful += 1
        else:
            self.failed += 1
            error_type = result.name
            self.errors[error_type] = self.errors.get(error_type, 0) + 1

    def clear(self) -> None:
        """Reset all statistics."""
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.errors.clear()
        self.last_update = None


class WordRow(TypedDict):
    """Structure of a word row from the database."""

    id: int
    term: str
    definition: str
    usage_examples: str


class Embedder(Protocol):
    """Protocol for text embedding generators."""

    def embed(self, text: str) -> np.ndarray:
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
    """Protocol defining the required interface for vector workers."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_status(self) -> VectorWorkerStatus: ...
    def is_alive(self) -> bool: ...


@final
class VectorWorker(threading.Thread):
    """
    Continuously scans the DB for words, generates embeddings, and stores them.

    This worker runs as a daemon thread that polls a database at regular intervals,
    generates vector embeddings for each word (combining term, definition, and usage examples),
    and stores these embeddings in a vector store for similarity search.
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
            poll_interval: Time in seconds between database polling cycles (defaults to config)
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
    """

    def __init__(self, dimension: int = 768):
        """
        Initialize with specified embedding dimension.

        Args:
            dimension: Size of the embedding vector
        """
        self.dimension = dimension

    def embed(self, text: str) -> np.ndarray:
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

            self.model = SentenceTransformer(model_name)
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

    def embed(self, text: str) -> np.ndarray:
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
            return self.model.encode(
                formatted_text, convert_to_numpy=True, normalize_embeddings=True
            )
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


def run_vector_worker_example() -> None:
    """
    Run an example of the VectorWorker with sample data.

    This creates a temporary database with sample words, initializes
    the vector worker components, and runs the worker for a brief period.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("vector_worker_example")
    logger.info("Starting vector worker example")

    # Define paths
    db_path = Path("./temp_example.db")
    vector_store_path = Path("./temp_vector_store")

    try:
        # Create temporary database with sample data
        with temporary_database(db_path) as db_file:
            # Initialize components
            db_manager = DBManager(db_path=str(db_file))

            # Try to use TransformerEmbedder, fall back to SimpleEmbedder if not available
            try:
                embedder = TransformerEmbedder()
                dimension = embedder.dimension  # 1024 for transformer model
                logger.info(f"Using TransformerEmbedder with dimension {dimension}")
            except Exception as e:
                dimension = 768  # Default for SimpleEmbedder
                embedder = SimpleEmbedder(dimension=dimension)
                logger.info(
                    f"Falling back to SimpleEmbedder with dimension {dimension}: {str(e)}"
                )

            vector_store = VectorStore(
                dimension=dimension,
                index_path=str(vector_store_path),
                storage_type=StorageType.MEMORY,
            )

            # Create and start worker
            worker = VectorWorker(
                db=db_manager,
                vector_store=vector_store,
                embedder=embedder,
                poll_interval=5.0,  # Short interval for demo
                logger=logger,
            )

            logger.info("Starting vector worker")
            worker.start()

            # Let it run for a short time
            run_duration = 10  # Reduced for demonstration
            logger.info(f"Worker will run for {run_duration} seconds")
            time.sleep(run_duration)

            # Stop the worker
            logger.info("Stopping vector worker")
            worker.stop()
            worker.join(timeout=2)

            if worker.is_alive():
                logger.warning("Worker did not terminate gracefully within timeout")

            logger.info("Example complete")

    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
    finally:
        # Clean up any leftover files
        if db_path.exists():
            db_path.unlink()
            logger.info(f"Removed temporary database: {db_path}")

        if vector_store_path.exists() and vector_store_path.is_dir():
            import shutil

            shutil.rmtree(vector_store_path)
            logger.info(f"Removed temporary vector store: {vector_store_path}")


if __name__ == "__main__":
    run_vector_worker_example()
