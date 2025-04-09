import sqlite3
from contextlib import contextmanager
from typing import Iterator, List, Optional, cast  # Import Iterator

# Import Result and Error types correctly
from word_forge.configs.config_essentials import (
    Error,
    ErrorCategory,
    ErrorSeverity,
    Result,
)

# Import protocols and types from conversation_types
from word_forge.conversation.conversation_types import (
    AffectiveLexicalModel,
    ConversationDict,
    IdentityModel,
    LightweightModel,
    MessageDict,
    ModelContext,
    ReflexiveModel,  # Import ReflexiveModel protocol
)
from word_forge.database.database_manager import DatabaseError, DBManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.vectorizer.vector_store import VectorStore


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
ORDER BY cm.timestamp ASC -- Order by timestamp for chronological order
"""


class ConversationManager:
    """
    Manages conversation sessions and messages, integrating multi-model response generation.

    Orchestrates a multi-stage response generation pipeline involving:
    1. Reflexive Model: Quick initial response/context.
    2. Lightweight Model: Routing and basic processing.
    3. Affective/Lexical Model: Core understanding and response generation.
    4. Identity Model: Personality, consistency, and final refinement.

    Provides methods for starting, ending, adding messages to, and retrieving
    conversations, ensuring data persistence and interaction flow management.
    """

    def __init__(
        self,
        db_manager: DBManager,
        emotion_manager: EmotionManager,
        graph_manager: GraphManager,
        vector_store: VectorStore,
        reflexive_model: ReflexiveModel,
        lightweight_model: LightweightModel,
        affective_model: AffectiveLexicalModel,
        identity_model: IdentityModel,
    ) -> None:
        """
        Initialize the conversation manager with dependencies and models.

        Args:
            db_manager: Database manager instance for persistence.
            emotion_manager: Emotion manager instance for analysis.
            graph_manager: Graph manager instance for knowledge access.
            vector_store: Vector store instance for similarity searches.
            reflexive_model: The initial reflexive model instance.
            lightweight_model: The routing/basic processing model instance.
            affective_model: The core understanding and response model instance.
            identity_model: The personality and refinement model instance.

        Raises:
            ConversationError: If initialization fails (e.g., table creation).
        """
        self.db_manager = db_manager
        self.emotion_manager = emotion_manager
        self.graph_manager = graph_manager
        self.vector_store = vector_store
        self.reflexive_model = reflexive_model
        self.lightweight_model = lightweight_model
        self.affective_model = affective_model
        self.identity_model = identity_model
        self._create_tables()  # Ensure tables are created on init

    @contextmanager
    def _db_connection(self) -> Iterator[sqlite3.Connection]:
        """
        Provides a managed database connection context.

        Ensures the connection uses a Row factory for dictionary-like access
        and handles connection acquisition and release through the DBManager.

        Yields:
            sqlite3.Connection: An active database connection.

        Raises:
            ConversationError: If obtaining a connection fails.
        """
        conn: Optional[sqlite3.Connection] = None
        try:
            with self.db_manager.get_connection() as db_conn:
                conn = db_conn
                conn.row_factory = sqlite3.Row
                yield conn
        except DatabaseError as e:
            raise ConversationError(f"Failed to get database connection: {e}") from e
        except Exception as e:
            raise ConversationError(
                f"Unexpected error during database connection: {e}"
            ) from e

    def _create_tables(self) -> None:
        """
        Ensures necessary database tables for conversations and messages exist.

        Executes CREATE TABLE IF NOT EXISTS statements for `conversations`
        and `conversation_messages`.

        Raises:
            ConversationError: If there's an issue executing the SQL statements.
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
        Initiates a new conversation record in the database.

        Sets the initial status to 'ACTIVE' and records the creation timestamp.

        Returns:
            int: The unique ID assigned to the newly created conversation.

        Raises:
            ConversationError: If the database insertion fails or the ID cannot be retrieved.
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
        Marks an existing conversation as 'COMPLETED'.

        Updates the conversation's status and `updated_at` timestamp.

        Args:
            conversation_id: The ID of the conversation to mark as completed.

        Raises:
            ConversationNotFoundError: If no conversation with the given ID exists.
            ConversationError: If the database update operation fails.
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
        Adds a message to a conversation and optionally triggers response generation.

        Persists the message to the database, updates the conversation's timestamp,
        processes the message for emotion, and if requested, initiates the
        multi-model response generation pipeline for an assistant reply.

        Args:
            conversation_id: The ID of the target conversation.
            speaker: The identifier of the message sender (e.g., "User", "Assistant").
            text: The textual content of the message. Cannot be empty or whitespace.
            generate_response: If True and the speaker is not "Assistant", triggers
                               the response generation pipeline. Defaults to False.

        Returns:
            int: The unique ID of the newly added message (the input message, not the
                 potential assistant response).

        Raises:
            ValueError: If the provided message text is empty or whitespace.
            ConversationNotFoundError: If the specified conversation_id does not exist
                                      (can be raised during response generation).
            ConversationError: If database operations fail or the response generation
                               pipeline encounters an unrecoverable error.
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
                if generate_response and speaker.lower() != "assistant":
                    print(
                        f"Triggering response generation for conversation {conversation_id}..."
                    )
                    response_result: Result[int] = self.generate_and_add_response(
                        conversation_id, text, speaker
                    )
                    if response_result.is_failure:
                        error_details = "Unknown error during response generation"
                        if response_result.error:
                            error_details = response_result.error.message
                        print(
                            f"Error generating response for conversation {conversation_id}: {error_details}"
                        )
                        raise ConversationError(
                            f"Failed to generate response: {error_details}"
                        )
                    else:
                        print(
                            f"Successfully generated and added response (ID: {response_result.unwrap()})"
                        )

                return message_id

        except sqlite3.Error as e:
            raise ConversationError(
                f"Failed to add message to conversation {conversation_id}: {e}"
            ) from e
        except Exception as e:
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
        Generates and adds an assistant response using the multi-model pipeline.

        Orchestrates the flow: Context Prep -> Reflexive -> Lightweight ->
        Affective/Lexical -> Identity -> Add Response. Handles errors at each stage.

        Args:
            conversation_id: ID of the current conversation.
            last_user_text: Text of the last message triggering the response.
            last_user_speaker: Speaker of the last message triggering the response.

        Returns:
            Result[int]: Contains the ID of the added assistant message on success,
                         or an Error object detailing the failure on error.
        """
        try:
            # --- 1. Prepare Context ---
            print(f"[{conversation_id}] Preparing context...")
            conv_result = self.get_conversation_if_exists(conversation_id)
            if not conv_result:
                return Result[int].failure(
                    Error.create(
                        "CONVERSATION_NOT_FOUND",
                        f"Conversation {conversation_id} not found during response generation.",
                        ErrorCategory.VALIDATION,
                        ErrorSeverity.ERROR,
                        {"conversation_id": conversation_id},
                    )
                )
            conversation_data = conv_result

            context: ModelContext = {
                "conversation_id": conversation_id,
                "history": conversation_data["messages"][-20:],
                "current_input": last_user_text,
                "speaker": last_user_speaker,
                "db_manager": self.db_manager,
                "emotion_manager": self.emotion_manager,
                "graph_manager": self.graph_manager,
                "vector_store": self.vector_store,
                "reflexive_output": None,
                "intermediate_response": None,
                "affective_state": None,
                "identity_state": None,
                "additional_data": {},
            }
            print(f"[{conversation_id}] Context prepared.")

            # --- 2. Reflexive Model ---
            print(f"[{conversation_id}] Calling Reflexive Model...")
            reflexive_result = self.reflexive_model.generate_reflex(context)
            if reflexive_result.is_failure:
                print(
                    f"[{conversation_id}] Warning: Reflexive model failed: {reflexive_result.error.message if reflexive_result.error else 'Unknown'}. Proceeding without reflex."
                )
            else:
                context = reflexive_result.unwrap()
                print(f"[{conversation_id}] Reflexive Model finished.")

            # --- 3. Lightweight Model ---
            print(f"[{conversation_id}] Calling Lightweight Model...")
            lightweight_result = self.lightweight_model.process(context)
            if lightweight_result.is_failure:
                print(
                    f"[{conversation_id}] Error: Lightweight model failed: {lightweight_result.error.message if lightweight_result.error else 'Unknown'}"
                )
                return Result[int].failure(lightweight_result.error)
            context = lightweight_result.unwrap()
            print(f"[{conversation_id}] Lightweight Model finished.")

            # --- 4. Affective/Lexical Model ---
            print(f"[{conversation_id}] Calling Affective/Lexical Model...")
            affective_result = self.affective_model.generate_core_response(context)
            if affective_result.is_failure:
                print(
                    f"[{conversation_id}] Error: Affective model failed: {affective_result.error.message if affective_result.error else 'Unknown'}"
                )
                return Result[int].failure(affective_result.error)
            context = affective_result.unwrap()
            print(
                f"[{conversation_id}] Affective Model generated intermediate response: '{context.get('intermediate_response', '')[:50]}...'"
            )

            # --- 5. Identity Model ---
            print(f"[{conversation_id}] Calling Identity Model...")
            identity_result = self.identity_model.refine_response(context)
            if identity_result.is_failure:
                print(
                    f"[{conversation_id}] Error: Identity model failed: {identity_result.error.message if identity_result.error else 'Unknown'}"
                )
                return Result[int].failure(identity_result.error)
            final_response_text = identity_result.unwrap()
            print(
                f"[{conversation_id}] Identity Model produced final response: '{final_response_text[:50]}...'"
            )

            # --- 6. Add Assistant Response ---
            print(f"[{conversation_id}] Adding final assistant response to DB...")
            assistant_message_id = self._add_assistant_message_internal(
                conversation_id, final_response_text
            )
            print(
                f"[{conversation_id}] Assistant message {assistant_message_id} added."
            )
            return Result[int].success(assistant_message_id)

        except Exception as e:
            print(
                f"[{conversation_id}] Unexpected error during response generation pipeline: {type(e).__name__} - {e}"
            )
            error = Error.create(
                "RESPONSE_PIPELINE_ERROR",
                f"Unexpected error in response pipeline: {e}",
                ErrorCategory.UNEXPECTED,
                ErrorSeverity.ERROR,
                {
                    "conversation_id": conversation_id,
                    "exception_type": type(e).__name__,
                },
            )
            return Result[int].failure(error)

    def _add_assistant_message_internal(self, conversation_id: int, text: str) -> int:
        """
        Internal helper to add an 'Assistant' message without recursion.

        Handles database insertion, timestamp update, and emotion processing
        for the assistant's message.

        Args:
            conversation_id: The ID of the conversation.
            text: The assistant's message text.

        Returns:
            int: The ID of the newly added assistant message.

        Raises:
            ConversationError: If the database operation fails.
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(SQL_ADD_MESSAGE, (conversation_id, "Assistant", text))
                message_id = cursor.lastrowid
                cursor.execute(SQL_UPDATE_CONVERSATION_TIMESTAMP, (conversation_id,))
                conn.commit()
                if message_id is None:
                    raise ConversationError("Failed to retrieve assistant message ID.")
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
        Retrieves full conversation details, including messages and emotions.

        Fetches the conversation metadata and all associated messages, joining
        with emotion data where available. Messages are ordered chronologically.

        Args:
            conversation_id: The ID of the conversation to retrieve.

        Returns:
            ConversationDict: A dictionary representing the conversation and its messages.

        Raises:
            ConversationNotFoundError: If no conversation with the given ID exists.
            ConversationError: If a database error occurs during retrieval.
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(SQL_GET_CONVERSATION, (conversation_id,))
                row: Optional[sqlite3.Row] = cursor.fetchone()

                if not row:
                    raise ConversationNotFoundError(
                        f"Conversation with ID {conversation_id} not found"
                    )

                conv_data: ConversationDict = {
                    "id": int(cast(int, row["id"])),
                    "status": str(cast(str, row["status"])),
                    "created_at": float(cast(float, row["created_at"])),
                    "updated_at": float(cast(float, row["updated_at"])),
                    "messages": [],
                }

                cursor.execute(SQL_GET_MESSAGES, (conversation_id,))
                messages: List[sqlite3.Row] = cursor.fetchall()

                for m_row in messages:
                    message: MessageDict = {
                        "id": int(cast(int, m_row["id"])),
                        "speaker": str(cast(str, m_row["speaker"])),
                        "text": str(cast(str, m_row["text"])),
                        "timestamp": float(cast(float, m_row["timestamp"])),
                        "emotion": None,
                    }

                    emotion_label = m_row["emotion_label"]
                    emotion_confidence = m_row["emotion_confidence"]
                    if emotion_label is not None and emotion_confidence is not None:
                        try:
                            confidence_float = float(emotion_confidence)
                            message["emotion"] = {
                                "emotion_label": str(emotion_label),
                                "confidence": confidence_float,
                            }
                        except (ValueError, TypeError):
                            print(
                                f"Warning: Invalid emotion data for message {message['id']} - Label: {emotion_label}, Confidence: {emotion_confidence}"
                            )

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
        Retrieves conversation details only if the conversation exists.

        A convenience wrapper around `get_conversation` that returns None
        instead of raising `ConversationNotFoundError`.

        Args:
            conversation_id: The ID of the conversation to attempt retrieval for.

        Returns:
            Optional[ConversationDict]: The conversation data if found, otherwise None.

        Raises:
            ConversationError: If a database error occurs (other than not found).
        """
        try:
            return self.get_conversation(conversation_id)
        except ConversationNotFoundError:
            return None
        except ConversationError as e:
            raise e
