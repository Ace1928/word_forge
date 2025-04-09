import sqlite3
from contextlib import contextmanager
from typing import List, Optional, cast  # Keep cast for sqlite3.Row

from word_forge.configs.config_essentials import (  # Import Result and Error types
    ErrorCategory,
    ErrorSeverity,
    Result,
)
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
from word_forge.database.database_manager import DatabaseError, DBManager
from word_forge.emotion.emotion_manager import EmotionManager
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
SELECT cm.id, cm.speaker, cm.text, cm.timestamp, me.label as emotion_label, me.confidence as emotion_confidence
FROM conversation_messages cm
LEFT JOIN message_emotion me ON cm.id = me.message_id
WHERE cm.conversation_id = ?
ORDER BY cm.id ASC
"""


class ConversationManager:
    """
    Manages conversation sessions and messages, integrating multi-model response generation.

    This class provides an interface for storing and retrieving conversation data,
    including conversation metadata and the messages within each conversation.
    It orchestrates a multi-stage response generation pipeline involving lightweight,
    affective/lexical, and identity models.
    """

    def __init__(
        self,
        db_manager: DBManager,
        emotion_manager: EmotionManager,
        graph_manager: GraphManager,
        vector_store: VectorStore,
        lightweight_model: LightweightModel,
        affective_model: AffectiveLexicalModel,
        identity_model: IdentityModel,
    ) -> None:
        """
        Initialize the conversation manager with dependencies and models.

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
        self.graph_manager = graph_manager
        self.vector_store = vector_store
        self.lightweight_model = lightweight_model
        self.affective_model = affective_model
        self.identity_model = identity_model
        self._create_tables()

    @contextmanager
    def _db_connection(self):
        """Create a database connection using the DBManager's path.

        Yields:
            sqlite3.Connection: An active database connection with Row factory.
        """
        conn: Optional[sqlite3.Connection] = None
        try:
            # Use DBManager's context manager for connection handling
            with self.db_manager.get_connection() as db_conn:
                conn = db_conn  # Assign to outer scope variable
                conn.row_factory = sqlite3.Row  # Ensure row factory is set
                yield conn
        except DatabaseError as e:
            raise ConversationError(f"Failed to get database connection: {e}") from e
        except Exception as e:
            raise ConversationError(
                f"Unexpected error during database connection: {e}"
            ) from e
        # No finally needed as DBManager's context manager handles closing

    def _create_tables(self) -> None:
        """
        Create tables if they don't already exist.

        Raises:
            ConversationError: If there's an issue creating the tables.
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
            The ID of the newly created conversation.

        Raises:
            ConversationError: If the database operation fails.
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
                print(
                    f"Started new conversation with ID: {conversation_id}"
                )  # Added log
                return conversation_id
        except sqlite3.Error as e:
            raise ConversationError(f"Failed to start new conversation: {e}") from e

    def end_conversation(self, conversation_id: int) -> None:
        """
        Marks a conversation as COMPLETED.

        Args:
            conversation_id: The ID of the conversation to end.

        Raises:
            ConversationNotFoundError: If the conversation doesn't exist.
            ConversationError: If the database operation fails.
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
                print(f"Ended conversation with ID: {conversation_id}")  # Added log
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
            generate_response: If True and speaker is not 'Assistant', trigger the
                               response generation pipeline.

        Returns:
            The ID of the newly added message (the user's message, not the potential response).

        Raises:
            ConversationError: If the database operation fails or response generation fails.
        """
        if not text or not text.strip():
            raise ValueError("Message text cannot be empty.")

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
                print(
                    f"Added message {message_id} from {speaker} to conversation {conversation_id}"
                )  # Added log

                # Process emotion for the added message (can happen after commit)
                try:
                    self.emotion_manager.process_message(message_id, text)
                except Exception as emotion_e:
                    # Log emotion processing error but don't fail message addition
                    print(
                        f"Warning: Failed to process emotion for message {message_id}: {emotion_e}"
                    )

                # Generate and add response if requested and not from Assistant
                if generate_response and speaker != "Assistant":
                    print(
                        f"Triggering response generation for conversation {conversation_id}..."
                    )  # Added log
                    response_result = self.generate_and_add_response(
                        conversation_id, text, speaker
                    )
                    if response_result.is_failure:
                        # Log the error and raise a ConversationError
                        error_details = (
                            response_result.error.message
                            if response_result.error
                            else "Unknown error"
                        )
                        print(
                            f"Error generating response for conversation {conversation_id}: {error_details}"
                        )  # Added log
                        raise ConversationError(
                            f"Failed to generate response: {error_details}"
                        )
                    else:
                        print(
                            f"Successfully generated and added response (ID: {response_result.unwrap()})"
                        )  # Added log

                return message_id  # Return the ID of the original message added

        except sqlite3.Error as e:
            raise ConversationError(
                f"Failed to add message to conversation {conversation_id}: {e}"
            ) from e
        except Exception as e:  # Catch other potential errors like response generation
            # Log the specific error
            print(
                f"Error in add_message for conversation {conversation_id}: {type(e).__name__} - {e}"
            )
            raise ConversationError(
                f"Error during message addition or response generation for conversation {conversation_id}: {e}"
            ) from e

    def generate_and_add_response(
        self, conversation_id: int, last_user_text: str, last_user_speaker: str
    ) -> Result[int]:
        """
        Generates a response using the multi-model pipeline and adds it to the conversation.

        Orchestrates the flow through Lightweight, Affective/Lexical, and Identity models.

        Args:
            conversation_id: The ID of the current conversation.
            last_user_text: The text of the last user message that triggered this response.
            last_user_speaker: The speaker of the last user message.

        Returns:
            Result containing the ID of the added assistant message or an error detailing
            which stage of the pipeline failed.
        """
        try:
            # 1. Prepare Context
            # Fetch full conversation data, including messages with emotions
            conversation_data = self.get_conversation(conversation_id)
            if not conversation_data:  # Should not happen if called after add_message
                return Result.failure(
                    "CONVERSATION_NOT_FOUND",
                    f"Conversation {conversation_id} not found during response generation.",
                )

            context: ModelContext = {
                "conversation_id": conversation_id,
                "history": conversation_data["messages"],  # Use fetched history
                "current_input": last_user_text,
                "speaker": last_user_speaker,
                "db_manager": self.db_manager,
                "emotion_manager": self.emotion_manager,
                "graph_manager": self.graph_manager,
                "vector_store": self.vector_store,
                "intermediate_response": None,
                "affective_state": None,
                "identity_state": None,  # Will be populated if IdentityModel uses state
            }
            print(f"Context prepared for conversation {conversation_id}")  # Added log

            # --- Model Pipeline ---
            # 2. Lightweight Model Processing
            print("Calling Lightweight Model...")  # Added log
            lightweight_result = self.lightweight_model.process(context)
            if lightweight_result.is_failure:
                return Result.failure(
                    "LIGHTWEIGHT_MODEL_ERROR",
                    f"Lightweight model failed: {lightweight_result.error.message if lightweight_result.error else 'Unknown'}",
                    (
                        lightweight_result.error.context
                        if lightweight_result.error
                        else {}
                    ),
                )
            context = lightweight_result.unwrap()
            print("Lightweight Model finished.")  # Added log

            # 3. Affective/Lexical Model Processing
            print("Calling Affective/Lexical Model...")  # Added log
            affective_result = self.affective_model.generate_core_response(context)
            if affective_result.is_failure:
                return Result.failure(
                    "AFFECTIVE_MODEL_ERROR",
                    f"Affective model failed: {affective_result.error.message if affective_result.error else 'Unknown'}",
                    affective_result.error.context if affective_result.error else {},
                )
            context = affective_result.unwrap()
            print(
                f"Affective Model generated intermediate response: '{context.get('intermediate_response', '')[:50]}...'"
            )  # Added log

            # 4. Identity Model Processing
            print("Calling Identity Model...")  # Added log
            identity_result = self.identity_model.refine_response(context)
            if identity_result.is_failure:
                return Result.failure(
                    "IDENTITY_MODEL_ERROR",
                    f"Identity model failed: {identity_result.error.message if identity_result.error else 'Unknown'}",
                    identity_result.error.context if identity_result.error else {},
                )
            final_response_text = identity_result.unwrap()
            print(
                f"Identity Model produced final response: '{final_response_text[:50]}...'"
            )  # Added log
            # --- End Model Pipeline ---

            # 5. Add Assistant Response (use internal method to avoid recursion)
            assistant_message_id = self._add_assistant_message_internal(
                conversation_id, final_response_text
            )
            return Result.success(assistant_message_id)

        except Exception as e:
            # Catch any unexpected errors during the pipeline execution
            print(
                f"Unexpected error during response generation pipeline: {type(e).__name__} - {e}"
            )  # Added log
            return Result.failure(
                "RESPONSE_GENERATION_PIPELINE_ERROR",
                f"Unexpected error during response generation pipeline: {e}",
                category=ErrorCategory.UNEXPECTED,
                severity=ErrorSeverity.ERROR,
            )

    def _add_assistant_message_internal(self, conversation_id: int, text: str) -> int:
        """Internal method to add an assistant message without triggering response generation."""
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(SQL_ADD_MESSAGE, (conversation_id, "Assistant", text))
                message_id = cursor.lastrowid
                cursor.execute(SQL_UPDATE_CONVERSATION_TIMESTAMP, (conversation_id,))
                conn.commit()
                if message_id is None:
                    raise ConversationError("Failed to retrieve assistant message ID.")
                # Process emotion for the assistant's message
                try:
                    self.emotion_manager.process_message(message_id, text)
                except Exception as emotion_e:
                    print(
                        f"Warning: Failed to process emotion for assistant message {message_id}: {emotion_e}"
                    )
                return message_id
        except sqlite3.Error as e:
            raise ConversationError(f"Failed to add assistant message: {e}") from e

    def get_conversation(self, conversation_id: int) -> ConversationDict:
        """
        Returns conversation info, including messages with emotions, if found.

        Args:
            conversation_id: The ID of the conversation to retrieve.

        Returns:
            Dictionary containing conversation data and messages.

        Raises:
            ConversationNotFoundError: If the conversation doesn't exist.
            ConversationError: If the database operation fails.
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

                # Use cast for type safety with sqlite3.Row
                conv_data: ConversationDict = {
                    "id": int(cast(int, row["id"])),
                    "status": str(cast(str, row["status"])),
                    "created_at": float(cast(float, row["created_at"])),
                    "updated_at": float(cast(float, row["updated_at"])),
                    "messages": [],
                }

                # Fetch messages with emotion data using LEFT JOIN
                cursor.execute(SQL_GET_MESSAGES, (conversation_id,))
                messages: List[sqlite3.Row] = cursor.fetchall()

                for m in messages:
                    message: MessageDict = {
                        "id": int(cast(int, m["id"])),
                        "speaker": str(cast(str, m["speaker"])),
                        "text": str(cast(str, m["text"])),
                        "timestamp": float(cast(float, m["timestamp"])),
                        "emotion": None,  # Default to None
                    }

                    # Populate emotion if data exists from the JOIN
                    emotion_label = m["emotion_label"]
                    emotion_confidence = m["emotion_confidence"]
                    if emotion_label is not None and emotion_confidence is not None:
                        message["emotion"] = {
                            "emotion_label": str(emotion_label),
                            "confidence": float(emotion_confidence),
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
            conversation_id: The ID of the conversation to retrieve.

        Returns:
            Dictionary containing conversation data or None if not found.

        Raises:
            ConversationError: If the database operation fails (excluding NotFound).
        """
        try:
            return self.get_conversation(conversation_id)
        except ConversationNotFoundError:
            return None
        except ConversationError as e:
            # Re-raise other conversation errors
            raise e
