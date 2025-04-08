import sqlite3
import time
from contextlib import contextmanager
from typing import List, Optional, TypedDict, cast

from word_forge.database.database_manager import (  # type:ignore[import]
    DatabaseError,
    DBManager,
)
from word_forge.emotion.emotion_manager import (  # type:ignore[import]
    EmotionAnalysisDict,
    EmotionManager,
)


class ConversationError(DatabaseError):
    """Base exception for conversation operations."""

    pass


class ConversationNotFoundError(ConversationError):
    """Raised when a conversation cannot be found."""

    pass


class MessageDict(TypedDict):
    """Type definition for message data structure."""

    id: int
    speaker: str
    text: str
    timestamp: float
    emotion: Optional[EmotionAnalysisDict]


class ConversationDict(TypedDict):
    """Type definition for conversation data structure."""

    id: int
    status: str
    created_at: float
    updated_at: float
    messages: List[MessageDict]


# SQL query constants to prevent duplication and enhance maintainability
SQL_CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT DEFAULT 'ACTIVE',
    created_at REAL DEFAULT (strftime('%s','now')),
    updated_at REAL DEFAULT (strftime('%s','now'))
)
"""

SQL_CREATE_CONVERSATION_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    speaker TEXT,
    text TEXT,
    timestamp REAL DEFAULT (strftime('%s','now')),
    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
)
"""

SQL_START_CONVERSATION = """
INSERT INTO conversations (status) VALUES ('ACTIVE')
"""

SQL_END_CONVERSATION = """
UPDATE conversations
SET status = 'COMPLETED',
    updated_at = strftime('%s','now')
WHERE id = ?
"""

SQL_ADD_MESSAGE = """
INSERT INTO conversation_messages (conversation_id, speaker, text)
VALUES (?, ?, ?)
"""

SQL_UPDATE_CONVERSATION_TIMESTAMP = """
UPDATE conversations
SET updated_at = strftime('%s','now')
WHERE id = ?
"""

SQL_GET_CONVERSATION = """
SELECT id, status, created_at, updated_at
FROM conversations
WHERE id = ?
"""

SQL_GET_MESSAGES = """
SELECT id, speaker, text, timestamp
FROM conversation_messages
WHERE conversation_id = ?
ORDER BY id ASC
"""


class ConversationManager:
    """
    Manages conversation sessions and messages.

    This class provides an interface for storing and retrieving conversation data,
    including conversation metadata and the messages within each conversation.
    Optionally integrates with EmotionManager for sentiment analysis.
    """

    def __init__(
        self, db_manager: DBManager, emotion_manager: Optional[EmotionManager] = None
    ) -> None:
        """
        Initialize the conversation manager with database and emotion managers.

        Args:
            db_manager: Database manager providing connection to storage
            emotion_manager: Optional emotion manager for sentiment analysis

        Raises:
            ConversationError: If there's an issue initializing the tables
        """
        self.db_manager = db_manager
        self.emotion_manager = emotion_manager
        self._create_tables()

    @contextmanager
    def _db_connection(self):
        """Create a database connection using the DBManager's path.

        Yields:
            sqlite3.Connection: An active database connection
        """
        conn = sqlite3.connect(self.db_manager.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _create_tables(self) -> None:
        """
        Create tables if they don't already exist:
          conversations(id, status, created_at, updated_at)
          conversation_messages(id, conversation_id, speaker, text, timestamp)

        Raises:
            ConversationError: If there's an issue creating the tables
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(SQL_CREATE_CONVERSATIONS_TABLE)
                cursor.execute(SQL_CREATE_CONVERSATION_MESSAGES_TABLE)
                conn.commit()
        except sqlite3.Error as e:
            raise ConversationError(
                f"Failed to initialize conversation tables: {e}"
            ) from e

    def start_conversation(self) -> int:
        """
        Creates a new conversation row and returns its ID.

        Returns:
            The ID of the newly created conversation

        Raises:
            ConversationError: If the database operation fails
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(SQL_START_CONVERSATION)
                conversation_id = cursor.lastrowid
                conn.commit()
                return cast(int, conversation_id)
        except sqlite3.Error as e:
            raise ConversationError(f"Failed to start new conversation: {e}") from e

    def end_conversation(self, conversation_id: int) -> None:
        """
        Marks a conversation as COMPLETED.

        Args:
            conversation_id: The ID of the conversation to end

        Raises:
            ConversationError: If the database operation fails
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(SQL_END_CONVERSATION, (conversation_id,))
                conn.commit()
                if cursor.rowcount == 0:
                    raise ConversationNotFoundError(
                        f"Conversation with ID {conversation_id} not found"
                    )
        except sqlite3.Error as e:
            raise ConversationError(
                f"Failed to end conversation {conversation_id}: {e}"
            ) from e

    def add_message(self, conversation_id: int, speaker: str, text: str) -> int:
        """
        Appends a message to the specified conversation.

        If an emotion manager is available, the message's emotional content
        will be analyzed and stored.

        Args:
            conversation_id: The ID of the conversation
            speaker: The name or identifier of the message sender
            text: The content of the message

        Returns:
            The ID of the newly added message

        Raises:
            ConversationError: If the database operation fails
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(SQL_ADD_MESSAGE, (conversation_id, speaker, text))
                message_id = cursor.lastrowid

                # Update conversation timestamp
                cursor.execute(SQL_UPDATE_CONVERSATION_TIMESTAMP, (conversation_id,))
                conn.commit()

                # Process emotion if manager is available
                if self.emotion_manager and message_id:
                    self.emotion_manager.process_message(message_id, text)

                return cast(int, message_id)
        except sqlite3.Error as e:
            raise ConversationError(
                f"Failed to add message to conversation {conversation_id}: {e}"
            ) from e

    def get_conversation(self, conversation_id: int) -> ConversationDict:
        """
        Returns conversation info, including messages, if found.

        If an emotion manager is available, each message will include its
        emotional analysis data.

        Args:
            conversation_id: The ID of the conversation to retrieve

        Returns:
            Dictionary containing conversation data and messages

        Raises:
            ConversationNotFoundError: If the conversation doesn't exist
            ConversationError: If the database operation fails
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                # Fetch conversation row
                cursor.execute(SQL_GET_CONVERSATION, (conversation_id,))
                row = cursor.fetchone()

                if not row:
                    raise ConversationNotFoundError(
                        f"Conversation with ID {conversation_id} not found"
                    )

                conv_data: ConversationDict = {
                    "id": row["id"],
                    "status": row["status"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "messages": [],
                }

                # Fetch messages
                cursor.execute(SQL_GET_MESSAGES, (conversation_id,))
                messages = cursor.fetchall()

                for m in messages:
                    message: MessageDict = {
                        "id": m["id"],
                        "speaker": m["speaker"],
                        "text": m["text"],
                        "timestamp": m["timestamp"],
                        "emotion": None,
                    }

                    # Fetch emotion data if available
                    if self.emotion_manager:
                        emotion_data = self.emotion_manager.get_message_emotion(m["id"])
                        if emotion_data:
                            message["emotion"] = {
                                "emotion_label": emotion_data["label"],
                                "confidence": emotion_data["confidence"],
                            }

                    conv_data["messages"].append(message)

                return conv_data

        except sqlite3.Error as e:
            raise ConversationError(
                f"Database error retrieving conversation {conversation_id}: {e}"
            ) from e

    def get_conversation_if_exists(
        self, conversation_id: int
    ) -> Optional[ConversationDict]:
        """
        Returns conversation info if it exists, None otherwise.

        Args:
            conversation_id: The ID of the conversation to retrieve

        Returns:
            Dictionary containing conversation data or None if not found

        Raises:
            ConversationError: If the database operation fails
        """
        try:
            return self.get_conversation(conversation_id)
        except ConversationNotFoundError:
            return None


def main() -> None:
    """
    Demonstrate the usage of ConversationManager with a complete workflow example.
    """

    # Initialize database manager
    db_path = "word_forge.sqlite"
    db_manager = DBManager(db_path)
    print(f"Using database at {db_path}")

    # Initialize emotion manager for sentiment analysis
    emotion_manager = EmotionManager(db_manager)

    # Initialize conversation manager with emotion integration
    conversation_manager = ConversationManager(db_manager, emotion_manager)
    print("Conversation manager initialized with emotion analysis capabilities")

    # Start a new conversation
    conversation_id = conversation_manager.start_conversation()
    print(f"Started new conversation with ID: {conversation_id}")

    # Add messages to the conversation
    messages = [
        ("User", "Hello! Can you help me understand what machine learning is?"),
        (
            "Assistant",
            "Of course! Machine learning is a branch of artificial intelligence focused on building systems that learn from data.",
        ),
        ("User", "That sounds interesting! Can you give me a simple example?"),
        (
            "Assistant",
            "Sure! A common example is email spam detection. The system learns patterns from emails marked as spam to identify new spam messages.",
        ),
        ("User", "Thanks, that makes sense! I appreciate your help."),
        (
            "Assistant",
            "You're welcome! I'm glad I could help. Feel free to ask if you have more questions.",
        ),
    ]

    print("\n=== Adding messages to conversation ===")
    for speaker, text in messages:
        message_id = conversation_manager.add_message(conversation_id, speaker, text)
        print(f"Added message from {speaker} (ID: {message_id})")

    # Retrieve and display the conversation
    print("\n=== Conversation Transcript ===")
    conversation = conversation_manager.get_conversation(conversation_id)

    print(f"Conversation ID: {conversation['id']}")
    print(f"Status: {conversation['status']}")
    print(f"Created: {time.ctime(conversation['created_at'])}")
    print(f"Last updated: {time.ctime(conversation['updated_at'])}")
    print(f"Message count: {len(conversation['messages'])}")

    print("\n=== Messages with Emotion Analysis ===")
    for msg in conversation["messages"]:
        print(f"\n{msg['speaker']} ({time.ctime(msg['timestamp'])}):")
        print(f"  \"{msg['text']}\"")

        if msg["emotion"]:
            emotion = msg["emotion"]["emotion_label"]
            confidence = msg["emotion"]["confidence"]
            print(f"  Emotion: {emotion} (confidence: {confidence:.2f})")

    # End the conversation
    conversation_manager.end_conversation(conversation_id)
    print("\nConversation marked as COMPLETED")

    # Verify the status change
    updated_conversation = conversation_manager.get_conversation(conversation_id)
    print(f"Final status: {updated_conversation['status']}")

    # Demonstrate error handling with a non-existent conversation
    non_existent_id = 9999
    print(f"\nTrying to retrieve non-existent conversation (ID: {non_existent_id})...")
    try:
        conversation_manager.get_conversation(non_existent_id)
    except ConversationNotFoundError as e:
        print(f"Error (as expected): {e}")

    # Use the safe lookup method
    result = conversation_manager.get_conversation_if_exists(non_existent_id)
    print(f"Safe lookup result: {'Found' if result else 'Not found'}")

    print("\nDemo complete!")


if __name__ == "__main__":
    main()
