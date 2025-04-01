import logging
import threading
import time
import traceback
from enum import Enum, auto
from typing import Optional, Protocol, TypedDict, final

from word_forge.conversation.conversation_manager import ConversationManager
from word_forge.parser.parser_refiner import ParserRefiner
from word_forge.queue.queue_manager import QueueManager


class ConversationState(Enum):
    """Worker lifecycle states for monitoring and control."""

    RUNNING = auto()
    STOPPED = auto()
    ERROR = auto()

    def __str__(self) -> str:
        """Return lowercase state name for consistent string representation."""
        return self.name.lower()


class ConversationError(Exception):
    """Base exception for conversation processing errors."""

    pass


class ConversationTaskError(ConversationError):
    """Raised when conversation task processing fails."""

    pass


class ConversationDBError(ConversationError):
    """Raised when database operations for conversations fail."""

    pass


class ConversationWorkerStatus(TypedDict):
    """Type definition for worker status information."""

    running: bool
    processed_count: int
    error_count: int
    last_update: Optional[float]
    uptime: Optional[float]
    state: str


class ConversationTask(TypedDict):
    """Type definition for conversation tasks in the queue."""

    task_type: str
    conversation_id: int
    message_text: str


class ConversationWorkerInterface(Protocol):
    """Protocol defining the required interface for conversation workers."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_status(self) -> ConversationWorkerStatus: ...
    def is_alive(self) -> bool: ...


@final
class ConversationWorker(threading.Thread):
    """
    Processes conversation tasks in the background (e.g., generating replies).
    For each queued item, it retrieves conversation context and uses the ParserRefiner
    or other LLM logic to generate a response, then saves it.
    """

    def __init__(
        self,
        conversation_manager: ConversationManager,
        parser_refiner: ParserRefiner,
        queue_manager: QueueManager,
        sleep_interval: Optional[float] = None,
        daemon: bool = True,
    ):
        """
        Initialize the conversation worker thread.

        Args:
            conversation_manager: Manager for conversation operations
            parser_refiner: Manager for text processing operations
            queue_manager: Queue system for conversation tasks
            sleep_interval: Seconds to wait when queue is empty (defaults to config)
            daemon: Whether thread should run as daemon
        """
        super().__init__(daemon=daemon)
        self.conversation_manager = conversation_manager
        self.parser_refiner = parser_refiner
        self.queue_manager = queue_manager
        self.sleep_interval = sleep_interval or 1.0

        self._stop_flag = False
        self._current_state = ConversationState.STOPPED
        self._status_lock = threading.RLock()
        self._start_time: Optional[float] = None
        self._last_update: Optional[float] = None
        self._processed_count = 0
        self._error_count = 0
        self.logger = logging.getLogger(__name__)

    def run(self) -> None:
        """
        Main execution loop that processes conversation tasks until stopped.

        Continuously polls the queue for tasks, processes them, and handles
        any exceptions during processing.
        """
        with self._status_lock:
            self._start_time = time.time()
            self._current_state = ConversationState.RUNNING

        self.logger.info("ConversationWorker started")

        while not self._stop_flag:
            try:
                # We'll store tasks in the queue as (task_type, conversation_id, message_text)
                task = self.queue_manager.dequeue_word()
                if task:
                    if isinstance(task, tuple) and len(task) == 3:
                        task_type, conv_id, message_text = task
                        if task_type == "conversation_message":
                            self._process_conversation_message(conv_id, message_text)
                    else:
                        # If the queue item is not recognized, skip
                        self.logger.warning(
                            f"Skipping unrecognized task format: {task}"
                        )
                else:
                    time.sleep(self.sleep_interval)
            except Exception as e:
                self._handle_processing_error(e)
                time.sleep(max(1.0, self.sleep_interval / 2))

        with self._status_lock:
            self._current_state = ConversationState.STOPPED

        self.logger.info("ConversationWorker stopped")

    def stop(self) -> None:
        """Signal the worker thread to stop gracefully."""
        self.logger.info("ConversationWorker stopping...")
        self._stop_flag = True

    def get_status(self) -> ConversationWorkerStatus:
        """
        Return the current status of the conversation worker.

        Returns:
            Dictionary containing operational metrics including:
            - running: Whether the worker is active
            - processed_count: Number of successfully processed messages
            - error_count: Number of encountered errors
            - last_update: Timestamp of last successful update
            - uptime: Seconds since thread start if running
            - state: Current worker state as string
        """
        with self._status_lock:
            uptime = None
            if self._start_time:
                uptime = time.time() - self._start_time

            status: ConversationWorkerStatus = {
                "running": self.is_alive() and not self._stop_flag,
                "processed_count": self._processed_count,
                "error_count": self._error_count,
                "last_update": self._last_update,
                "uptime": uptime,
                "state": str(self._current_state),
            }

            return status

    def _process_conversation_message(
        self, conversation_id: int, user_text: str
    ) -> None:
        """
        Process a user message by generating and saving a system reply.

        Args:
            conversation_id: ID of the conversation to process
            user_text: Text of the user's message

        Raises:
            ConversationTaskError: If message processing fails
        """
        try:
            self.conversation_manager.add_message(conversation_id, "USER", user_text)

            # Retrieve entire conversation for context
            conv_data = self.conversation_manager.get_conversation(conversation_id)
            if not conv_data:
                raise ConversationTaskError(f"Conversation {conversation_id} not found")

            # Generate a reply to the user's message
            system_reply = self._generate_reply(user_text)

            # Save system reply
            self.conversation_manager.add_message(
                conversation_id, "SYSTEM", system_reply
            )

            with self._status_lock:
                self._processed_count += 1
                self._last_update = time.time()

            self.logger.debug(f"Processed message in conversation {conversation_id}")

        except Exception as e:
            self.logger.error(f"Error processing conversation message: {str(e)}")
            traceback.print_exc()
            raise ConversationTaskError(f"Failed to process message: {str(e)}") from e

    def _generate_reply(self, user_text: str) -> str:
        """
        Generate a reply to the user's message based on its content.

        Args:
            user_text: The text to respond to

        Returns:
            Generated reply text
        """
        # For a more advanced approach, integrate a conversation LLM here
        if "hello" in user_text.lower():
            return "Hello! How can I help you today?"
        else:
            # Possibly do some lexical expansions or synonyms
            # For now, let's just echo or do something simple
            return f"I heard you say: '{user_text}'. That's quite interesting!"

    def _handle_processing_error(self, error: Exception) -> None:
        """
        Log processing errors with contextual information and update state.

        Args:
            error: The exception that occurred
        """
        with self._status_lock:
            self._error_count += 1
            self._current_state = ConversationState.ERROR

        error_type = type(error).__name__
        self.logger.error(f"{error_type}: {str(error)}")
        self.logger.debug(traceback.format_exc())
