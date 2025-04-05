"""
Vector Embedding Demonstration for Word Forge.

This module demonstrates the complete vector embedding workflow from text to search,
showcasing the multilingual capabilities and semantic search features of the system.
It serves as both a functional example and a testing tool for the vector components.

Usage:
    python -m word_forge.vectorizer.vector_demo

Architecture:
    The demo creates a self-contained environment with:
    1. In-memory or disk-based SQLite database
    2. Vector embeddings using selected model
    3. Search functionality across multiple languages
"""

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, cast

from word_forge.database.db_manager import DBManager, WordEntryDict
from word_forge.vectorizer.vector_store import (
    SearchResultDict,
    StorageType,
    VectorStore,
)
from word_forge.vectorizer.vector_worker import SimpleEmbedder, TransformerEmbedder

# Type definitions
WordID = int
Language = Literal["en", "fr", "zh", "es", "de", "ja"]
EmbedderType = Union[SimpleEmbedder, TransformerEmbedder]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vector_demo")


@dataclass
class SearchResult:
    """
    Formatted search result for display purposes.

    Structured container for search result data that maintains
    all information in a display-ready format.

    Attributes:
        term: The word or phrase that matched
        content_type: Type of content ("word", "definition", "example", etc.)
        language: Language code of the content
        distance: Semantic distance from query (lower is better)
        text: The actual text content that matched
        text_preview: Truncated version of text for display
    """

    term: str
    content_type: str
    language: str
    distance: float
    text: Optional[str] = None
    text_preview: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """
        Generate text preview if text is available.

        Truncates long text to 80 characters with an ellipsis,
        or leaves short text unchanged.
        """
        if self.text:
            self.text_preview = (
                f"{self.text[:80]}..." if len(self.text) > 80 else self.text
            )


class DatabaseSetupError(Exception):
    """
    Raised when database initialization fails.

    Occurs when the SQLite database cannot be created or
    configured properly with the necessary schema.
    """

    pass


class EmbedderInitializationError(Exception):
    """
    Raised when embedding model initialization fails.

    Occurs when neither the transformer model nor the fallback
    simple embedder can be successfully initialized.
    """

    pass


class VectorStoreInitializationError(Exception):
    """
    Raised when vector store initialization fails.

    Occurs when the vector storage system cannot be properly
    configured for storing and retrieving embeddings.
    """

    pass


class WordStorageError(Exception):
    """
    Raised when storing a word in the database or vector store fails.

    Occurs when database operations or vector embedding generation
    encounter errors during word storage processing.
    """

    pass


class VectorDemo:
    """
    Demonstration of Word Forge vector embedding and search capabilities.

    This class orchestrates the complete lifecycle of vector operations:
    1. Setting up a test database
    2. Initializing embedding models and vector storage
    3. Processing words into embeddings
    4. Performing various types of semantic searches
    5. Demonstrating multilingual capabilities

    Attributes:
        db_manager: Database manager for word storage
        vector_store: Vector store for embeddings
        embedder: Embedding generator
        words: Dictionary of test words by ID
        demo_path: Path to store demonstration files
        db_path: Path to SQLite database file
        vector_path: Path to vector store files
    """

    def __init__(
        self,
        use_transformer: bool = True,
        storage_type: StorageType = StorageType.MEMORY,
        demo_path: Optional[Path] = None,
    ):
        """
        Initialize the vector demonstration.

        Creates a complete vector embedding environment with database,
        embedding model, and vector store for demonstration purposes.

        Args:
            use_transformer: Whether to use the transformer model (True) or simple embedder (False)
            storage_type: Type of storage to use (memory or disk)
            demo_path: Optional path for storing demonstration files

        Raises:
            DatabaseSetupError: If database initialization fails
            EmbedderInitializationError: If embedding model initialization fails
            VectorStoreInitializationError: If vector store initialization fails
        """
        # Initialize paths
        self.demo_path = demo_path or Path("./vector_demo")
        self.demo_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.demo_path / "demo.sqlite"
        self.vector_path = self.demo_path / "vectors"

        # Initialize components
        try:
            self._setup_database()
        except Exception as e:
            raise DatabaseSetupError(f"Failed to initialize database: {str(e)}") from e

        try:
            self.embedder = self._initialize_embedder(use_transformer)
        except Exception as e:
            raise EmbedderInitializationError(
                f"Failed to initialize embedder: {str(e)}"
            ) from e

        try:
            self.vector_store = self._initialize_vector_store(storage_type)
        except Exception as e:
            raise VectorStoreInitializationError(
                f"Failed to initialize vector store: {str(e)}"
            ) from e

        # Keep track of added words
        self.words: Dict[WordID, WordEntryDict] = {}

    def _setup_database(self) -> None:
        """
        Set up a test database with schema for demonstration.

        Creates an SQLite database with the necessary tables for
        storing word entries with their definitions and usage examples.

        Raises:
            sqlite3.Error: If database operations fail
            ConnectionError: If database connection cannot be established
        """
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Create words table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS words (
                    id INTEGER PRIMARY KEY,
                    term TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    usage_examples TEXT,
                    language TEXT
                )
                """
            )

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

        finally:
            if conn:
                conn.close()

        # Initialize DB manager
        self.db_manager = DBManager(db_path=str(self.db_path))

    def _initialize_embedder(self, use_transformer: bool) -> EmbedderType:
        """
        Initialize the appropriate embedding model.

        Attempts to create a transformer-based embedder if specified,
        with fallback to the simpler model if that fails.

        Args:
            use_transformer: Whether to use the transformer model

        Returns:
            Initialized embedder instance

        Raises:
            Exception: If both transformer and fallback initialization fail
        """
        if use_transformer:
            try:
                embedder = TransformerEmbedder()
                logger.info(
                    f"Using TransformerEmbedder with dimension {embedder.dimension}"
                )
                return embedder
            except Exception as e:
                logger.warning(f"Failed to initialize transformer embedder: {e}")
                logger.info("Falling back to SimpleEmbedder")
                return SimpleEmbedder()

        # If not using transformer or fallback case
        embedder = SimpleEmbedder()
        logger.info(f"Using SimpleEmbedder with dimension {embedder.dimension}")
        return embedder

    def _initialize_vector_store(self, storage_type: StorageType) -> VectorStore:
        """
        Initialize the vector store with the appropriate configuration.

        Creates a vector store configured to work with the selected
        embedder and storage type.

        Args:
            storage_type: Type of storage to use (memory or disk)

        Returns:
            Initialized vector store
        """
        return VectorStore(
            dimension=self.embedder.dimension,
            index_path=str(self.vector_path),
            storage_type=storage_type,
            db_manager=self.db_manager,
        )

    def add_word(
        self,
        term: str,
        definition: str,
        usage_examples: Union[str, List[str]] = "",
        language: Language = "en",
    ) -> WordID:
        """
        Add a word to the database and process it for vector storage.

        Creates a database entry for the word and generates vector
        embeddings for semantic search capabilities.

        Args:
            term: The word or phrase
            definition: The meaning or explanation
            usage_examples: Examples of the word in context (string or list)
            language: Language code (en, fr, zh, es, de, ja)

        Returns:
            ID of the added word

        Raises:
            WordStorageError: If storing the word fails

        Examples:
            >>> demo = VectorDemo()
            >>> word_id = demo.add_word(
            ...     "algorithm",
            ...     "A step-by-step procedure for solving a problem",
            ...     ["Sorting algorithms organize data efficiently"],
            ...     "en"
            ... )
        """
        # Format usage examples
        examples_str = self._format_usage_examples(usage_examples)

        # Get next ID
        next_id = len(self.words) + 1

        # Create word entry
        word: WordEntryDict = {
            "id": next_id,
            "term": term,
            "definition": definition,
            "usage_examples": examples_str,
            "language": language,
            "part_of_speech": "",  # Required by WordEntryDict
            "last_refreshed": 0.0,  # Required by WordEntryDict
            "relationships": [],  # Required by WordEntryDict
        }

        try:
            # Store in database using transaction
            with self.db_manager.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO words (id, term, definition, usage_examples, language)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (next_id, term, definition, examples_str, language),
                )

            # Process for vector storage
            vectors_added = self.vector_store.store_word(word)

            # Keep track of added word
            self.words[next_id] = word

            logger.info(
                f"Added word '{term}' (ID: {next_id}, Language: {language}) with {vectors_added} vectors"
            )
            return next_id

        except Exception as e:
            raise WordStorageError(f"Failed to store word '{term}': {str(e)}") from e

    def _format_usage_examples(self, usage_examples: Union[str, List[str]]) -> str:
        """
        Format usage examples into a standard string format.

        Converts list of examples to semicolon-separated string format
        for consistent database storage.

        Args:
            usage_examples: String or list of usage examples

        Returns:
            Semicolon-separated string of examples
        """
        if isinstance(usage_examples, list):
            return "; ".join(usage_examples)
        return usage_examples

    def add_multilingual_examples(self) -> None:
        """
        Add a set of multilingual test examples to the demonstration.

        Creates a standard set of words in English, Chinese, and French
        to demonstrate the multilingual capabilities of the system.

        Raises:
            WordStorageError: If adding any word fails
        """
        logger.info("Adding multilingual examples...")

        # Dictionary of examples by language
        examples: Dict[Language, List[Dict[str, Union[str, List[str]]]]] = {
            "en": [
                {
                    "term": "algorithm",
                    "definition": "A step-by-step procedure for solving a problem or accomplishing a task.",
                    "usage_examples": [
                        "The sorting algorithm efficiently organized the data.",
                        "Computer scientists developed a new algorithm for image recognition.",
                    ],
                },
                {
                    "term": "recursion",
                    "definition": "The process of defining something in terms of itself.",
                    "usage_examples": [
                        "The function uses recursion to calculate factorial numbers.",
                        "Recursion is a powerful technique in programming.",
                    ],
                },
            ],
            "zh": [
                {
                    "term": "算法",  # algorithm
                    "definition": "解决问题或完成任务的一步一步的程序。",
                    "usage_examples": [
                        "排序算法高效地组织数据。",
                        "计算机科学家开发了一种新的图像识别算法。",
                    ],
                },
                {
                    "term": "递归",  # recursion
                    "definition": "一种通过自身定义的过程。",
                    "usage_examples": [
                        "该函数使用递归计算阶乘。",
                        "递归是编程中一种强大的技术。",
                    ],
                },
            ],
            "fr": [
                {
                    "term": "algorithme",
                    "definition": "Une procédure étape par étape pour résoudre un problème ou accomplir une tâche.",
                    "usage_examples": [
                        "L'algorithme de tri organise efficacement les données.",
                        "Les informaticiens ont développé un nouvel algorithme pour la reconnaissance d'images.",
                    ],
                },
                {
                    "term": "récursion",
                    "definition": "Le processus de définition de quelque chose en termes de soi-même.",
                    "usage_examples": [
                        "La fonction utilise la récursion pour calculer les factorielles.",
                        "La récursion est une technique puissante en programmation.",
                    ],
                },
            ],
        }

        # Add all examples
        for language, words in examples.items():
            for word in words:
                self.add_word(
                    term=cast(str, word["term"]),
                    definition=cast(str, word["definition"]),
                    usage_examples=word["usage_examples"],
                    language=language,
                )

        logger.info(f"Added {len(self.words)} multilingual examples")

    def search_similar(self, query: str, k: int = 3) -> List[SearchResult]:
        """
        Search for words similar to the query and display results.

        Performs semantic search using the query text and formats
        the results into a structured display.

        Args:
            query: Text query to search for
            k: Number of results to return

        Returns:
            List of formatted search results

        Examples:
            >>> demo = VectorDemo()
            >>> demo.add_multilingual_examples()
            >>> results = demo.search_similar("computer algorithms")
            Found 3 results:
            1. algorithm (Type: word, Language: en, Distance: 0.1234)
               Text: algorithm - A step-by-step procedure for solving a problem...
        """
        logger.info(f"Searching for: '{query}'")
        raw_results = self.vector_store.search(query_text=query, k=k)

        # Format results
        results = self._format_search_results(raw_results)

        # Display results
        self._display_search_results(query, results)

        return results

    def _format_search_results(
        self, raw_results: List[SearchResultDict]
    ) -> List[SearchResult]:
        """
        Convert raw search results into formatted search results.

        Transforms the raw data from the vector store into a more
        structured and display-friendly format.

        Args:
            raw_results: Results from vector store search

        Returns:
            List of formatted SearchResult objects
        """
        formatted_results: List[SearchResult] = []

        for result in raw_results:
            metadata = result["metadata"] or {}
            term = metadata.get("term", "Unknown")
            language = metadata.get("language", "Unknown")
            content_type = metadata.get("content_type", "Unknown")
            distance = result["distance"]

            formatted_result = SearchResult(
                term=str(term),
                content_type=content_type,
                language=str(language),
                distance=distance,
                text=result["text"],
            )

            formatted_results.append(formatted_result)

        return formatted_results

    def _display_search_results(self, query: str, results: List[SearchResult]) -> None:
        """
        Display formatted search results to the console.

        Prints the query and formatted results with proper spacing
        and formatting for easy reading.

        Args:
            query: The original search query
            results: Formatted search results to display
        """
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results)} results:")

        for i, result in enumerate(results):
            print(
                f"{i+1}. {result.term} (Type: {result.content_type}, "
                f"Language: {result.language}, Distance: {result.distance:.4f})"
            )

            # Show the text if available
            if result.text:
                print(f"   Text: {result.text_preview}")
            else:
                print("   Text: Not available")
        print("\n")


def main() -> None:
    """
    Main entry point for running the vector demo.

    Creates a demo instance, adds multilingual examples, and
    performs a sample search to showcase capabilities.
    """
    demo = VectorDemo()
    demo.add_multilingual_examples()

    # Perform sample searches
    demo.search_similar("algorithm for solving problems")
    demo.search_similar("递归技术", k=5)  # "recursive technique" in Chinese
    demo.search_similar(
        "procédure étape par étape"
    )  # "step by step procedure" in French


if __name__ == "__main__":
    main()
