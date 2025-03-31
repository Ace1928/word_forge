import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union, cast


class DatabaseError(Exception):
    """Base exception for database operations."""

    pass


class TermNotFoundError(DatabaseError):
    """Raised when a term cannot be found in the database."""

    pass


class RelationshipDict(TypedDict):
    """Type definition for relationship dictionary structure."""

    related_term: str
    relationship_type: str


class WordEntryDict(TypedDict):
    """Type definition for word entry dictionary structure."""

    id: int
    term: str
    definition: str
    part_of_speech: str
    usage_examples: List[str]
    last_refreshed: float
    relationships: List[RelationshipDict]


# SQL query constants to prevent duplication and enhance maintainability
SQL_CREATE_WORDS_TABLE = """
CREATE TABLE IF NOT EXISTS words (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    term TEXT UNIQUE NOT NULL,
    definition TEXT,
    part_of_speech TEXT,
    usage_examples TEXT,
    last_refreshed REAL
);
"""

SQL_CREATE_RELATIONSHIPS_TABLE = """
CREATE TABLE IF NOT EXISTS relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id INTEGER NOT NULL,
    related_term TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    FOREIGN KEY(word_id) REFERENCES words(id)
);
"""

SQL_CREATE_WORD_ID_INDEX = """
CREATE INDEX IF NOT EXISTS idx_relationships_word_id
ON relationships(word_id);
"""

SQL_CREATE_UNIQUE_RELATIONSHIP_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_relationship
ON relationships(word_id, related_term, relationship_type);
"""

SQL_INSERT_OR_UPDATE_WORD = """
INSERT INTO words (term, definition, part_of_speech, usage_examples, last_refreshed)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(term)
DO UPDATE SET
    definition=excluded.definition,
    part_of_speech=excluded.part_of_speech,
    usage_examples=excluded.usage_examples,
    last_refreshed=excluded.last_refreshed
"""

SQL_INSERT_RELATIONSHIP = """
INSERT OR IGNORE INTO relationships
(word_id, related_term, relationship_type)
VALUES (?, ?, ?)
"""

SQL_GET_WORD_ENTRY = """
SELECT id, term, definition, part_of_speech, usage_examples, last_refreshed
FROM words WHERE term = ?
"""

SQL_GET_RELATIONSHIPS = """
SELECT related_term, relationship_type
FROM relationships
WHERE word_id = ?
"""

SQL_GET_WORD_ID = """
SELECT id FROM words WHERE term = ?
"""


class DBManager:
    """
    Manages the SQLite database for terms, definitions, relationships, etc.

    This class provides an interface for storing and retrieving linguistic data,
    including words, definitions, and the relationships between words.
    """

    def __init__(self, db_path: Union[str, Path] = "word_forge.sqlite") -> None:
        """
        Initialize the database manager with path to SQLite database file.

        Args:
            db_path: Path to the SQLite database file (created if doesn't exist)
        """
        self.db_path: str = str(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        """
        Create the necessary database tables if they do not already exist.

        Creates:
            - words table: Stores word definitions and metadata
            - relationships table: Stores word relationships (synonyms, antonyms, etc.)

        Raises:
            DatabaseError: If there's an issue with database initialization
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Enable foreign keys for referential integrity
                cursor.execute("PRAGMA foreign_keys = ON;")

                # Create tables and indexes
                cursor.execute(SQL_CREATE_WORDS_TABLE)
                cursor.execute(SQL_CREATE_RELATIONSHIPS_TABLE)
                cursor.execute(SQL_CREATE_WORD_ID_INDEX)
                cursor.execute(SQL_CREATE_UNIQUE_RELATIONSHIP_INDEX)

                conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize database: {e}") from e

    def insert_or_update_word(
        self,
        term: str,
        definition: str = "",
        part_of_speech: str = "",
        usage_examples: Optional[List[str]] = None,
    ) -> None:
        """
        Insert or update a word's data in the database.

        Args:
            term: The textual term itself
            definition: The definition string
            part_of_speech: Part of speech (e.g., noun, verb, adjective)
            usage_examples: List of usage example sentences

        Raises:
            ValueError: If term is empty
            DatabaseError: If the database operation fails
        """
        if not term.strip():
            raise ValueError("Term cannot be empty")

        usage_str: str = "; ".join(usage_examples) if usage_examples else ""

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                timestamp: float = time.time()

                cursor.execute(
                    SQL_INSERT_OR_UPDATE_WORD,
                    (term, definition, part_of_speech, usage_str, timestamp),
                )
                conn.commit()

                # Clear cache when word is updated
                self._get_word_id.cache_clear()

        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to insert or update term '{term}': {e}") from e

    def insert_relationship(
        self, base_term: str, related_term: str, relationship_type: str
    ) -> bool:
        """
        Insert a relationship from base_term to related_term.

        Args:
            base_term: The source term
            related_term: The target term related to the base term
            relationship_type: The type of relationship (e.g., synonym, antonym)

        Returns:
            bool: True if the relationship was inserted, False if base_term doesn't exist

        Raises:
            ValueError: If any parameter is empty
            DatabaseError: If the database operation fails
        """
        self._validate_relationship_params(base_term, related_term, relationship_type)

        word_id = self._get_word_id(base_term)
        if word_id is None:
            return False  # Base term doesn't exist

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    SQL_INSERT_RELATIONSHIP,
                    (word_id, related_term, relationship_type),
                )
                conn.commit()
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to insert relationship from '{base_term}' to '{related_term}': {e}"
            ) from e

    def _validate_relationship_params(
        self, base_term: str, related_term: str, relationship_type: str
    ) -> None:
        """
        Validate relationship parameters.

        Args:
            base_term: The source term
            related_term: The target term related to the base term
            relationship_type: The type of relationship

        Raises:
            ValueError: If any parameter is empty
        """
        if not base_term.strip():
            raise ValueError("Base term cannot be empty")
        if not related_term.strip():
            raise ValueError("Related term cannot be empty")
        if not relationship_type.strip():
            raise ValueError("Relationship type cannot be empty")

    def get_word_entry(self, term: str) -> WordEntryDict:
        """
        Retrieve a word record from the database, along with any relationships.

        Args:
            term: The word term to look up

        Returns:
            A dictionary containing the word data and its relationships

        Raises:
            ValueError: If term is empty
            TermNotFoundError: If the term doesn't exist in the database
            DatabaseError: If the database operation fails
        """
        if not term.strip():
            raise ValueError("Term cannot be empty")

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get main word data
                cursor.execute(SQL_GET_WORD_ENTRY, (term,))
                row = cursor.fetchone()
                if not row:
                    raise TermNotFoundError(f"Term '{term}' not found in database")

                # Extract data from row
                word_entry = self._create_word_entry_from_row(dict(row))

                # Get relationships
                cursor.execute(SQL_GET_RELATIONSHIPS, (word_entry["id"],))
                relationships: List[RelationshipDict] = [
                    cast(
                        RelationshipDict,
                        {
                            "related_term": row["related_term"],
                            "relationship_type": row["relationship_type"],
                        },
                    )
                    for row in cursor.fetchall()
                ]

                word_entry["relationships"] = relationships
                return cast(WordEntryDict, word_entry)

        except sqlite3.Error as e:
            raise DatabaseError(f"Database error retrieving term '{term}': {e}") from e

    def _create_word_entry_from_row(self, row_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a word entry dictionary from a database row.

        Args:
            row_dict: Dictionary containing database row values

        Returns:
            Word entry dictionary without relationships
        """
        usage_str: str = row_dict["usage_examples"] or ""
        usage_examples: List[str] = usage_str.split("; ") if usage_str else []

        return {
            "id": row_dict["id"],
            "term": row_dict["term"],
            "definition": row_dict["definition"],
            "part_of_speech": row_dict["part_of_speech"],
            "usage_examples": usage_examples,
            "last_refreshed": row_dict["last_refreshed"],
            "relationships": [],  # Will be populated later
        }

    def get_word_if_exists(self, term: str) -> Optional[WordEntryDict]:
        """
        Retrieve a word record if it exists, returning None otherwise.

        Args:
            term: The word term to look up

        Returns:
            A dictionary containing the word data and its relationships, or None if not found

        Raises:
            DatabaseError: If the database operation fails
        """
        try:
            return self.get_word_entry(term)
        except (TermNotFoundError, ValueError):
            return None

    @lru_cache(maxsize=128)
    def _get_word_id(self, term: str) -> Optional[int]:
        """
        Retrieve the primary key ID for a given term.

        Args:
            term: The word term to look up

        Returns:
            The word ID if found, None otherwise

        Raises:
            DatabaseError: If the database operation fails
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(SQL_GET_WORD_ID, (term,))
                result = cursor.fetchone()
                return result[0] if result else None
        except sqlite3.Error as e:
            raise DatabaseError(
                f"Failed to retrieve word ID for term '{term}': {e}"
            ) from e

    def get_all_words(self) -> List[Dict[str, Any]]:
        """Return all words in the database with their data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT id, term, definition, usage_examples FROM words")
                return cursor.fetchall()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to retrieve all words: {e}") from e


def main() -> None:
    """
    Demonstrate the usage of DBManager with a complete workflow example.
    """
    import os
    import tempfile

    # Create a temporary database file
    temp_dir = tempfile.gettempdir()
    db_path = os.path.join(temp_dir, "word_forge_demo.sqlite")

    # Initialize the database manager
    db = DBManager(db_path)
    print(f"Database initialized at {db_path}")

    # Insert example words
    db.insert_or_update_word(
        term="algorithm",
        definition="A step-by-step procedure for solving a problem or accomplishing a task.",
        part_of_speech="noun",
        usage_examples=[
            "The search algorithm quickly found the matching records.",
            "Researchers developed a new algorithm for detecting patterns in large datasets.",
        ],
    )
    print("Inserted 'algorithm'")

    db.insert_or_update_word(
        term="recursion",
        definition="A programming concept where a function calls itself to solve smaller instances of the same problem.",
        part_of_speech="noun",
        usage_examples=["Recursion is often used to solve tree-based problems."],
    )
    print("Inserted 'recursion'")

    # Insert relationships
    result = db.insert_relationship(
        base_term="algorithm", related_term="procedure", relationship_type="synonym"
    )
    if result:
        print("Inserted 'algorithm' → 'procedure' relationship")

    result = db.insert_relationship(
        base_term="algorithm",
        related_term="recursion",
        relationship_type="related_concept",
    )
    if result:
        print("Inserted 'algorithm' → 'recursion' relationship")

    # Retrieve and display word entries
    try:
        algorithm_entry = db.get_word_entry("algorithm")
        print("\n=== Algorithm Entry ===")
        print(f"Term: {algorithm_entry['term']}")
        print(f"Definition: {algorithm_entry['definition']}")
        print(f"Part of Speech: {algorithm_entry['part_of_speech']}")
        print("Usage Examples:")
        for example in algorithm_entry["usage_examples"]:
            print(f"  - {example}")
        print("Relationships:")
        for rel in algorithm_entry["relationships"]:
            print(f"  - {rel['relationship_type']}: {rel['related_term']}")
    except TermNotFoundError:
        print("Term 'algorithm' not found")

    # Update an existing word
    db.insert_or_update_word(
        term="algorithm",
        definition="A defined set of step-by-step procedures that provides the correct answer to a particular problem.",
        part_of_speech="noun",
        usage_examples=[
            "The search algorithm quickly found the matching records.",
            "Researchers developed a new algorithm for detecting patterns in large datasets.",
            "The efficiency of an algorithm is measured by its time and space complexity.",
        ],
    )
    print("\nUpdated 'algorithm' entry")

    # Show the updated entry
    try:
        updated_entry = db.get_word_entry("algorithm")
        print("\n=== Updated Algorithm Entry ===")
        print(f"Definition: {updated_entry['definition']}")
        print(f"Usage Examples Count: {len(updated_entry['usage_examples'])}")
    except TermNotFoundError:
        print("Term 'algorithm' not found")

    # Demonstrate safe lookup for non-existent terms
    nonexistent = db.get_word_if_exists("nonexistent_term")
    print(
        f"\nNon-existent term lookup result: {'Found' if nonexistent else 'Not found'}"
    )

    print("\nDemo complete!")


# Export public elements
__all__ = [
    "DBManager",
    "WordEntryDict",
    "RelationshipDict",
    "DatabaseError",
    "TermNotFoundError",
]


if __name__ == "__main__":
    main()
