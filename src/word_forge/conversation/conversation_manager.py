import sqlite3
from contextlib import contextmanager
from typing import List, Optional  # Keep cast for now, Any for sqlite3.Row

from word_forge.configs.config_essentials import Result  # Import Result
from word_forge.conversation.conversation_models import (  # Import new protocols
    AffectiveLexicalModel,
    IdentityModel,
    LightweightModel,
    ModelContext,
)
from word_forge.conversation.conversation_types import (  # Import types from the new file
    ConversationDict,
    MessageDict,
)
from word_forge.database.database_manager import (  # type:ignore[import]
    DatabaseError,
    DBManager,
)
from word_forge.emotion.emotion_manager import EmotionManager  # type:ignore[import]
from word_forge.graph.graph_manager import GraphManager  # Import GraphManager
from word_forge.vectorizer.vector_store import VectorStore  # Import VectorStore


class ConversationError(DatabaseError):
    """Base exception for conversation operations."""

    pass


class ConversationNotFoundError(ConversationError):
    """Raised when a conversation cannot be found."""

    pass


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
    Manages conversation sessions and messages, integrating multi-model response generation.

    This class provides an interface for storing and retrieving conversation data,
    including conversation metadata and the messages within each conversation.
    Optionally integrates with EmotionManager for sentiment analysis.
    """

    def __init__(
        self,
        db_manager: DBManager,
        emotion_manager: EmotionManager,  # EmotionManager is now required
        graph_manager: GraphManager,  # Add GraphManager
        vector_store: VectorStore,  # Add VectorStore
        lightweight_model: LightweightModel,
        affective_model: AffectiveLexicalModel,
        identity_model: IdentityModel,
    ) -> None:
        """
        Initialize the conversation manager with database, emotion, graph, vector managers, and models.

        Args:
            db_manager: Database manager providing connection to storage.
            emotion_manager: Emotion manager for sentiment analysis.
            graph_manager: Graph manager for lexical relationships.
            vector_store: Vector store for semantic search.
            lightweight_model: The initial processing model.
            affective_model: The core understanding and response model.
            identity_model: The personality and refinement model.

        Raises:
            ConversationError: If there's an issue initializing the tables.
        """
        self.db_manager = db_manager
        self.emotion_manager = emotion_manager
        self.graph_manager = graph_manager  # Store GraphManager
        self.vector_store = vector_store  # Store VectorStore
        self.lightweight_model = lightweight_model
        self.affective_model = affective_model
        self.identity_model = identity_model
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
                if conversation_id is None:
                    raise ConversationError(
                        "Failed to retrieve conversation ID after insertion"
                    )
                return conversation_id
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

    def add_message(
        self,
        conversation_id: int,
        speaker: str,
        text: str,
        generate_response: bool = False,
    ) -> int:
        """
        Appends a message and optionally generates and adds an assistant response.

        Args:
            conversation_id: The ID of the conversation.
            speaker: The name or identifier of the message sender.
            text: The content of the message.
            generate_response: If True and speaker is not 'Assistant', generate and add a response.

        Returns:
            The ID of the newly added message (the user's message, not the potential response).

        Raises:
            ConversationError: If the database operation fails or response generation fails.
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(SQL_ADD_MESSAGE, (conversation_id, speaker, text))
                message_id = cursor.lastrowid

                # Update conversation timestamp
                cursor.execute(SQL_UPDATE_CONVERSATION_TIMESTAMP, (conversation_id,))
                conn.commit()

                if message_id is None:
                    raise ConversationError(
                        "Failed to retrieve message ID after insertion"
                    )

                # Process emotion for the added message
                self.emotion_manager.process_message(message_id, text)

                # Generate and add response if requested and not from Assistant
                if generate_response and speaker != "Assistant":
                    response_result = self.generate_and_add_response(
                        conversation_id, text, speaker
                    )
                    if response_result.is_failure:
                        # Propagate the error if response generation failed
                        raise ConversationError(
                            f"Failed to generate response: {response_result.error}"
                        )

                return message_id  # Return the ID of the original message added

        except sqlite3.Error as e:
            raise ConversationError(
                f"Failed to add message to conversation {conversation_id}: {e}"
            ) from e
        except Exception as e:  # Catch potential response generation errors
            raise ConversationError(
                f"Error during message addition or response generation for conversation {conversation_id}: {e}"
            ) from e

    def generate_and_add_response(
        self, conversation_id: int, last_user_text: str, last_user_speaker: str
    ) -> Result[int]:
        """
        Generates a response using the multi-model pipeline and adds it to the conversation.

        Args:
            conversation_id: The ID of the current conversation.
            last_user_text: The text of the last user message.
            last_user_speaker: The speaker of the last user message.

        Returns:
            Result containing the ID of the added assistant message or an error.
        """
        try:
            # 1. Prepare Context
            conversation_data = self.get_conversation(conversation_id)  # Fetch history
            context: ModelContext = {
                "conversation_id": conversation_id,
                "history": conversation_data["messages"],
                "current_input": last_user_text,
                "speaker": last_user_speaker,
                "db_manager": self.db_manager,
                "emotion_manager": self.emotion_manager,
                "graph_manager": self.graph_manager,
                "vector_store": self.vector_store,
                "intermediate_response": None,
                "affective_state": None,
                "identity_state": None,  # Placeholder for identity state
            }

            # 2. Lightweight Model Processing
            lightweight_result = self.lightweight_model.process(context)
            if lightweight_result.is_failure:
                return Result.failure(
                    "LIGHTWEIGHT_MODEL_ERROR",
                    f"Lightweight model failed: {lightweight_result.error}",
                )
            context = lightweight_result.unwrap()

            # 3. Affective/Lexical Model Processing
            affective_result = self.affective_model.generate_core_response(context)
            if affective_result.is_failure:
                return Result.failure(
                    "AFFECTIVE_MODEL_ERROR",
                    f"Affective model failed: {affective_result.error}",
                )
            context = affective_result.unwrap()

            # 4. Identity Model Processing
            identity_result = self.identity_model.refine_response(context)
            if identity_result.is_failure:
                return Result.failure(
                    "IDENTITY_MODEL_ERROR",
                    f"Identity model failed: {identity_result.error}",
                )
            final_response_text = identity_result.unwrap()

            # 5. Add Assistant Response (without triggering another response)
            assistant_message_id = self.add_message(
                conversation_id,
                "Assistant",
                final_response_text,
                generate_response=False,
            )
            return Result.success(assistant_message_id)

        except Exception as e:
            return Result.failure(
                "RESPONSE_GENERATION_ERROR", f"Failed to generate response: {e}"
            )

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
                row: Optional[sqlite3.Row] = cursor.fetchone()

                if not row:
                    raise ConversationNotFoundError(
                        f"Conversation with ID {conversation_id} not found"
                    )

                conv_data: ConversationDict = {
                    "id": int(row["id"]),
                    "status": str(row["status"]),
                    "created_at": float(row["created_at"]),
                    "updated_at": float(row["updated_at"]),
                    "messages": [],
                }

                # Fetch messages
                cursor.execute(SQL_GET_MESSAGES, (conversation_id,))
                messages: List[sqlite3.Row] = cursor.fetchall()

                for m in messages:
                    message: MessageDict = {
                        "id": int(m["id"]),
                        "speaker": str(m["speaker"]),
                        "text": str(m["text"]),
                        "timestamp": float(m["timestamp"]),
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
