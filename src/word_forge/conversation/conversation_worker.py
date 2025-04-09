"""
Conversation Worker Module for Word Forge.

This module provides a thread-based worker for continuous processing of
conversation tasks from a queue. It handles message generation, context tracking,
and asynchronous conversation management with comprehensive lifecycle control
and metrics tracking.

Architecture:
    ┌────────────────────┐
    │ ConversationWorker │
    └──────────┬─────────┘
               │
    ┌──────────┴──────────┐
    │     Components      │
    └─────────────────────┘
    ┌───────┬────────┬─────────┐
    │Queue  │Parser  │Conversa-│
    │Manager│Refiner │tionMgr  │
    └───────┴────────┴─────────┘
"""

from __future__ import annotations

import logging
import queue
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import Event, Lock, RLock, Thread
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
    cast,
    final,
)

from word_forge.conversation.conversation_manager import ConversationManager
from word_forge.database.database_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.parser.parser_refiner import ParserRefiner
from word_forge.queue.queue_manager import QueueManager, Result  # Import Result

# Configure module logger
logger = logging.getLogger(__name__)


class ConversationWorkerState(Enum):
    """Worker lifecycle states for monitoring and control."""

    RUNNING = auto()
    STOPPED = auto()
    ERROR = auto()
    PAUSED = auto()
    RECOVERY = auto()

    def __str__(self) -> str:
        """Return lowercase state name for consistent string representation."""
        return self.name.lower()


class ConversationError(Exception):
    """Base exception for conversation worker errors."""

    pass


class ConversationProcessingError(ConversationError):
    """Raised when processing a conversation task fails."""

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.cause = cause
        self.context = context or {}
        self.timestamp = time.time()

    def __str__(self) -> str:
        error_msg = str(self.args[0]) if self.args else ""
        if self.cause:
            error_msg += f" | Cause: {self.cause}"
        if self.context:
            error_msg += f" | Context: {self.context}"
        return error_msg


class ConversationTimeoutError(ConversationError):
    """Raised when a conversation task processing times out."""

    pass


class ConversationQueueError(ConversationError):
    """Raised when queue operations fail."""

    pass


class TaskResult(Enum):
    """Result status of a conversation task processing attempt."""

    SUCCESS = auto()
    FAILURE = auto()
    TIMEOUT = auto()
    DEFERRED = auto()
    INVALID = auto()


class ConversationWorkerStatus(TypedDict):
    """Type definition for worker status information."""

    running: bool
    processed_count: int
    success_count: int
    error_count: int
    last_update: Optional[float]
    uptime: Optional[float]
    state: str
    queue_size: int
    recent_errors: List[str]
    next_poll: Optional[float]
    conversation_metrics: Dict[str, Any]


class ConversationTask(TypedDict, total=False):
    """Type definition for a conversation task."""

    task_id: str
    conversation_id: Optional[int]
    message: str
    speaker: str
    timestamp: float
    priority: int
    context: Dict[str, Any]
    completion_callback: Optional[Callable[[Dict[str, Any]], None]]


class ConversationWorkerInterface(Protocol):
    """Protocol defining the required interface for conversation workers."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def restart(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def get_status(self) -> ConversationWorkerStatus: ...
    def get_metrics(self) -> Dict[str, Any]: ...  # Added get_metrics
    def is_alive(self) -> bool: ...
    def process_task(self, task: ConversationTask) -> bool: ...
    def submit_message(
        self, conversation_id: int, message: str, speaker: str
    ) -> Optional[str]: ...
    def run(self) -> None: ...


@dataclass
class ProcessingMetrics:
    """Tracks and aggregates processing metrics with thread safety."""

    processed_count: int = 0
    success_count: int = 0
    error_count: int = 0
    timeout_count: int = 0
    deferred_count: int = 0
    invalid_count: int = 0
    processing_times: List[float] = field(default_factory=list)
    task_types: Dict[str, int] = field(default_factory=lambda: {})
    error_types: Dict[str, int] = field(default_factory=lambda: {})
    _lock: Lock = field(default_factory=Lock)

    def record_processed(self) -> None:
        """Increment the processed count thread-safely."""
        with self._lock:
            self.processed_count += 1

    def record_result(self, result: TaskResult, processing_time: float) -> None:
        """Record a processing result with timing information."""
        with self._lock:
            if result == TaskResult.SUCCESS:
                self.success_count += 1
            elif result == TaskResult.FAILURE:
                self.error_count += 1
            elif result == TaskResult.TIMEOUT:
                self.timeout_count += 1
            elif result == TaskResult.DEFERRED:
                self.deferred_count += 1
            elif result == TaskResult.INVALID:
                self.invalid_count += 1

            self.processing_times.append(processing_time)
            # Keep only the last 100 processing times
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-100:]

    def record_task_type(self, task_type: str) -> None:
        """Record a task type for distribution analysis."""
        with self._lock:
            self.task_types[task_type] = self.task_types.get(task_type, 0) + 1

    def record_error(self, error_type: str) -> None:
        """Record an error type for error distribution analysis."""
        with self._lock:
            self.error_types[error_type] = self.error_types.get(error_type, 0) + 1

    def get_avg_processing_time(self) -> Optional[float]:
        """Calculate the average processing time with timeout protection."""
        try:
            # Use a timeout to prevent deadlocks during shutdown
            if not self._lock.acquire(timeout=0.5):
                return None

            try:
                if not self.processing_times:
                    return None
                return sum(self.processing_times) / len(self.processing_times)
            finally:
                self._lock.release()
        except Exception:
            # Safely handle any exceptions during shutdown
            return None

    def get_metrics_dict(self) -> Dict[str, Any]:
        """Return a dictionary of all metrics."""
        try:
            # Use a timeout to prevent deadlocks
            if not self._lock.acquire(timeout=1.0):
                # Return minimal metrics if we can't acquire the lock
                return {
                    "processed_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "success_rate": 0,
                    "error": "Metrics lock acquisition timed out",
                }

            try:
                # Calculate average processing time outside the lock
                avg_time = None
                if self.processing_times:
                    avg_time = sum(self.processing_times) / len(self.processing_times)

                return {
                    "processed_count": self.processed_count,
                    "success_count": self.success_count,
                    "error_count": self.error_count,
                    "timeout_count": self.timeout_count,
                    "deferred_count": self.deferred_count,
                    "invalid_count": self.invalid_count,
                    "avg_processing_time_ms": (
                        round(avg_time * 1000, 2) if avg_time is not None else None
                    ),
                    "task_type_distribution": dict(self.task_types),
                    "error_type_distribution": dict(self.error_types),
                    "success_rate": round(
                        self.success_count / max(1, self.processed_count) * 100, 2
                    ),
                }
            finally:
                self._lock.release()
        except Exception as e:
            # Return error metrics if something goes wrong
            return {"error": f"Failed to get metrics: {str(e)}", "processed_count": 0}


class StateTracker:
    """Manages worker state with thread-safe operations."""

    def __init__(self) -> None:
        """Initialize state tracker with default values."""
        self._lock = RLock()
        self._state = ConversationWorkerState.STOPPED
        self._start_time: Optional[float] = None
        self._last_update: Optional[float] = None
        self._next_poll: Optional[float] = None
        self._recent_errors: List[Tuple[float, str]] = []
        self._max_recent_errors = 10

    def start(self) -> None:
        """Transition to RUNNING state."""
        with self._lock:
            self._state = ConversationWorkerState.RUNNING
            self._start_time = time.time()
            self._last_update = time.time()

    def stop(self) -> None:
        """Transition to STOPPED state."""
        with self._lock:
            self._state = ConversationWorkerState.STOPPED
            self._last_update = time.time()

    def pause(self) -> None:
        """Transition to PAUSED state."""
        with self._lock:
            self._state = ConversationWorkerState.PAUSED
            self._last_update = time.time()

    def resume(self) -> None:
        """Transition to RUNNING state from PAUSED."""
        with self._lock:
            if self._state == ConversationWorkerState.PAUSED:
                self._state = ConversationWorkerState.RUNNING
                self._last_update = time.time()

    def error(self) -> None:
        """Transition to ERROR state."""
        with self._lock:
            self._state = ConversationWorkerState.ERROR
            self._last_update = time.time()

    def recovery(self) -> None:
        """Transition to RECOVERY state."""
        with self._lock:
            self._state = ConversationWorkerState.RECOVERY
            self._last_update = time.time()

    def record_update(self) -> None:
        """Record a successful update to the state."""
        with self._lock:
            self._last_update = time.time()

    def record_error(self, error_msg: str) -> None:
        """Record an error with timestamp."""
        with self._lock:
            self._recent_errors.append((time.time(), error_msg))
            # Keep only the most recent errors
            if len(self._recent_errors) > self._max_recent_errors:
                self._recent_errors = self._recent_errors[-self._max_recent_errors :]

    def set_next_poll(self, next_poll: float) -> None:
        """Set the next scheduled poll time."""
        with self._lock:
            self._next_poll = next_poll

    @property
    def state(self) -> ConversationWorkerState:
        """Get the current state."""
        with self._lock:
            return self._state

    @property
    def is_running(self) -> bool:
        """Check if the worker is in RUNNING state."""
        with self._lock:
            return self._state == ConversationWorkerState.RUNNING

    @property
    def uptime(self) -> Optional[float]:
        """Calculate uptime in seconds if started."""
        with self._lock:
            if self._start_time is None:
                return None
            return time.time() - self._start_time

    @property
    def last_update(self) -> Optional[float]:
        """Get timestamp of last update."""
        with self._lock:
            return self._last_update

    @property
    def next_poll(self) -> Optional[float]:
        """Get the next scheduled poll time."""
        with self._lock:
            return self._next_poll

    @property
    def recent_errors(self) -> List[str]:
        """Get list of recent error messages."""
        with self._lock:
            return [
                error
                for _, error in sorted(
                    self._recent_errors, key=lambda x: x[0], reverse=True
                )
            ]

    def to_dict(
        self, queue_size: int, metrics: Dict[str, Any]
    ) -> ConversationWorkerStatus:
        """Convert current state to status dictionary."""
        with self._lock:
            return ConversationWorkerStatus(
                running=self.is_running,
                state=str(self._state),
                processed_count=metrics["processed_count"],
                success_count=metrics["success_count"],
                error_count=metrics["error_count"],
                last_update=self._last_update,
                uptime=self.uptime,
                queue_size=queue_size,
                recent_errors=self.recent_errors[:5],
                next_poll=self._next_poll,
                conversation_metrics=metrics,
            )


@final
class ConversationWorker(Thread):
    """
    Thread-based worker for processing conversation tasks from a queue.

    Manages the lifecycle of conversation processing, including message generation,
    context tracking, and asynchronous conversation management. Implements
    comprehensive state management with start, stop, pause, resume functionality
    and detailed metrics tracking.

    Attributes:
        parser_refiner: Parser for lexical processing and response generation
        queue_manager: Queue system for conversation tasks
        conversation_manager: Manager for conversation persistence
        db_manager: Database access for word and message storage
        emotion_manager: Optional manager for emotional analysis
        metrics: Processing statistics and metrics
        state_tracker: Thread-safe state management
        poll_interval: Seconds between queue polling attempts
        processing_timeout: Maximum seconds to process a task
        _stop_event: Threading event to signal worker shutdown
        _pause_event: Threading event to signal worker pause
    """

    def __init__(
        self,
        parser_refiner: ParserRefiner,
        queue_manager: QueueManager[Union[ConversationTask, str, Dict[str, Any]]],
        conversation_manager: ConversationManager,
        db_manager: DBManager,
        emotion_manager: Optional[EmotionManager] = None,
        poll_interval: float = 2.0,
        processing_timeout: float = 30.0,
        batch_size: int = 1,
        max_retries: int = 3,
        daemon: bool = True,
        enable_logging: bool = True,
    ) -> None:
        """
        Initialize conversation worker with required components.

        Args:
            parser_refiner: Parser component for lexical processing
            queue_manager: Queue system for message retrieval
            conversation_manager: Manager for conversation persistence
            db_manager: Database access for storage
            emotion_manager: Optional emotion analysis component
            poll_interval: Seconds between queue polling attempts
            processing_timeout: Maximum seconds to process a task
            batch_size: Number of tasks to process per cycle
            max_retries: Maximum retry attempts for failed tasks
            daemon: Whether the thread should be a daemon
            enable_logging: Whether to log processing activity
        """
        super().__init__(name="ConversationWorker", daemon=daemon)
        self.parser_refiner = parser_refiner
        self.queue_manager = queue_manager
        self.conversation_manager = conversation_manager
        self.db_manager = db_manager
        self.emotion_manager = emotion_manager

        # Configuration
        self.poll_interval = poll_interval
        self.processing_timeout = processing_timeout
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.enable_logging = enable_logging

        # State management
        self._stop_event = Event()
        self._pause_event = Event()
        self.metrics = ProcessingMetrics()
        self.state_tracker = StateTracker()

        # Advanced tracking
        self._retry_queue: Dict[str, Tuple[ConversationTask, int]] = {}
        self._processing_times: Dict[str, float] = {}
        self._conversation_cache: Dict[int, Dict[str, Any]] = {}

        # Logger setup
        self.logger = logger if enable_logging else None
        if self.logger:
            self.logger.info("ConversationWorker initialized")

    def start(self) -> None:
        """
        Start the conversation worker thread.

        If the worker is already running, this method has no effect.
        """
        if self.is_alive():
            if self.logger:
                self.logger.warning("Worker thread already started")
            return

        self._stop_event.clear()
        self._pause_event.clear()
        self.state_tracker.start()

        if self.logger:
            self.logger.info("Starting conversation worker thread")

        super().start()

    def stop(self) -> None:
        """
        Signal the worker thread to stop processing and shut down.

        This method returns immediately, but the thread may continue
        until the current task is completed.
        """
        if not self.is_alive():
            return

        if self.logger:
            self.logger.info("Stopping conversation worker")

        self._stop_event.set()
        self._pause_event.clear()  # Ensure thread isn't paused when stopping
        self.state_tracker.stop()

    def restart(self) -> None:
        """
        Restart the worker thread by stopping it and starting a new one.

        If the thread is not alive, it will just be started.
        """
        if self.is_alive():
            self.stop()
            # Wait for thread to actually terminate
            self.join(timeout=min(self.processing_timeout, 10.0))

        # Reset internal state
        self._stop_event.clear()
        self._pause_event.clear()
        self._retry_queue.clear()
        self._processing_times.clear()

        # Start a new thread
        self.state_tracker.start()
        super().start()

        if self.logger:
            self.logger.info("Conversation worker restarted")

    def pause(self) -> None:
        """
        Pause the worker thread temporarily.

        The thread will finish processing the current task and then wait
        until resumed or stopped.
        """
        if not self.is_alive() or self._stop_event.is_set():
            return

        self._pause_event.set()
        self.state_tracker.pause()

        if self.logger:
            self.logger.info("Conversation worker paused")

    def resume(self) -> None:
        """
        Resume a paused worker thread.

        If the thread is not paused, this method has no effect.
        """
        if not self.is_alive() or self._stop_event.is_set():
            return

        self._pause_event.clear()
        self.state_tracker.resume()

        if self.logger:
            self.logger.info("Conversation worker resumed")

    def run(self) -> None:
        """
        Main execution loop for the worker thread.

        Continuously polls the queue for tasks and processes them until
        signaled to stop. Handles pausing, error recovery, and metrics tracking.
        """
        if self.logger:
            self.logger.info("Conversation worker thread started")

        last_error_recovery = 0.0
        backoff_factor = 1.0

        try:
            while not self._stop_event.is_set():
                # Handle pause state
                if self._pause_event.is_set():
                    time.sleep(0.5)  # Lighter sleep during pause
                    continue

                # Set next poll time for status reporting
                next_poll = time.time() + self.poll_interval
                self.state_tracker.set_next_poll(next_poll)

                try:
                    # Process retry queue first
                    self._process_retry_queue()

                    # Process new tasks from queue
                    processed = self._process_queue_batch()

                    if processed > 0:
                        # Reset backoff on successful processing
                        backoff_factor = 1.0
                        self.state_tracker.record_update()
                    else:
                        # Wait before next poll
                        time.sleep(self.poll_interval * backoff_factor)

                except Exception as e:
                    error_name = type(e).__name__
                    error_msg = f"{error_name}: {str(e)}"
                    self.state_tracker.record_error(error_msg)
                    self.metrics.record_error(error_name)

                    if self.logger:
                        self.logger.error(
                            f"Error in conversation worker: {error_msg}", exc_info=True
                        )

                    # Handle recovery
                    current_time = time.time()
                    if (
                        current_time - last_error_recovery > 60.0
                    ):  # 1 minute between recoveries
                        self.state_tracker.recovery()
                        if self.logger:
                            self.logger.info("Entering recovery mode")

                        # Recovery logic
                        self._clear_stale_processing()
                        last_error_recovery = current_time
                        self.state_tracker.resume()  # Back to running after recovery

                    # Exponential backoff capped at 60 seconds
                    backoff_factor = min(backoff_factor * 2, 60.0 / self.poll_interval)
                    time.sleep(self.poll_interval * backoff_factor)

        except Exception as e:
            if self.logger:
                self.logger.critical(
                    f"Critical error in conversation worker: {str(e)}", exc_info=True
                )
            self.state_tracker.error()

        finally:
            if self.logger:
                self.logger.info("Conversation worker thread terminated")

    def _process_retry_queue(self) -> int:
        """
        Process tasks in the retry queue.

        Returns:
            int: Number of tasks processed from the retry queue
        """
        if not self._retry_queue:
            return 0

        processed = 0
        retry_items = list(self._retry_queue.items())

        for task_id, (task, attempts) in retry_items:
            # Skip if we've reached max retries
            if attempts >= self.max_retries:
                if self.logger:
                    self.logger.warning(
                        f"Task {task_id} exceeded maximum retry attempts ({self.max_retries})"
                    )
                del self._retry_queue[task_id]
                continue

            # Process the retry
            try:
                result = self._process_task(task)
                processed += 1

                if result == TaskResult.SUCCESS:
                    # Task succeeded, remove from retry queue
                    del self._retry_queue[task_id]
                elif result == TaskResult.DEFERRED:
                    # Keep in retry queue but don't increment attempts
                    pass
                else:
                    # Increment retry attempt count
                    self._retry_queue[task_id] = (task, attempts + 1)

            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Error processing retry task {task_id}: {str(e)}",
                        exc_info=True,
                    )
                # Increment retry attempt count
                self._retry_queue[task_id] = (task, attempts + 1)

        return processed

    def _process_queue_batch(self) -> int:
        """Process a batch of tasks from the queue."""
        processed_count = 0

        for _ in range(self.batch_size):
            try:
                # Get a task result from the queue
                dequeue_result: Result[Union[ConversationTask, str, Dict[str, Any]]] = (
                    self.queue_manager.dequeue(block=False)
                )

                if dequeue_result.is_success:
                    task_item = dequeue_result.unwrap()

                    # Handle different types of queue items
                    if isinstance(task_item, dict) and "task_id" in task_item:
                        # This is a valid ConversationTask
                        task = cast(ConversationTask, task_item)
                        self._process_task(task)
                        processed_count += 1
                    elif isinstance(task_item, str):
                        # Handle string tasks by converting to simple task format
                        simple_task: ConversationTask = {
                            "task_id": f"auto_{int(time.time())}",
                            "conversation_id": None,
                            "message": task_item,
                            "speaker": "user",
                            "timestamp": time.time(),
                            "priority": 1,
                            "context": {},
                        }
                        self._process_task(simple_task)
                        processed_count += 1
                    else:
                        # Skip invalid task types with more informative logging
                        task_type = type(task_item).__name__
                        if self.logger:
                            self.logger.warning(
                                f"Invalid task type received: {task_type}. Expected ConversationTask. "
                                f"Item: {str(task_item)[:100]}... (truncated)"
                            )
                else:
                    # Handle error from dequeue_result
                    error = dequeue_result.error
                    if error is not None and error.error_code != "EMPTY_QUEUE":
                        if self.logger:
                            self.logger.error(f"Error dequeuing task: {error.message}")
                        self.state_tracker.record_error(
                            f"Dequeue error: {error.message}"
                        )

            except queue.Empty:
                # No more tasks in queue
                break
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Error processing queue batch: {e}", exc_info=True
                    )
                self.state_tracker.record_error(str(e))

        return processed_count

    def _process_task(self, task: ConversationTask) -> TaskResult:
        """
        Process a single conversation task.

        Args:
            task: The conversation task to process

        Returns:
            TaskResult: The result status of the processing attempt

        Raises:
            ConversationProcessingError: If task processing fails
        """
        task_id = task.get("task_id", str(id(task)))
        start_time = time.time()

        # Record task being processed
        self.metrics.record_processed()
        self._processing_times[task_id] = start_time

        try:
            # Validate task data
            message = task.get("message", "")
            if not message:
                if self.logger:
                    self.logger.warning(f"Task {task_id} has empty message")
                self.metrics.record_result(TaskResult.INVALID, time.time() - start_time)
                return TaskResult.INVALID

            # Extract task parameters
            conversation_id = task.get("conversation_id")
            speaker = task.get("speaker", "user")
            context = task.get("context", {})

            # Process based on task type (determine by context and existence of conversation_id)
            if conversation_id is None:
                # Create new conversation
                conversation_id = self.conversation_manager.start_conversation()
                if self.logger:
                    self.logger.info(f"Created new conversation {conversation_id}")

            # Add message to conversation
            message_id = self.conversation_manager.add_message(
                conversation_id, speaker, message
            )

            if self.logger:
                self.logger.debug(
                    f"Added message {message_id} to conversation {conversation_id}"
                )

            # Generate response if needed
            if context.get("generate_response", False) and speaker != "assistant":
                response = self._generate_response(conversation_id, message, context)

                if response:
                    # Add response to conversation
                    self.conversation_manager.add_message(
                        conversation_id, "assistant", response
                    )

                    if self.logger:
                        self.logger.info(
                            f"Generated response for conversation {conversation_id}"
                        )

            # Execute callback if provided
            completion_callback = task.get("completion_callback")
            if completion_callback and callable(completion_callback):
                result_data: Dict[str, Any] = {
                    "task_id": task_id,
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                    "processed_at": time.time(),
                    "success": True,
                }
                completion_callback(result_data)

            # Record successful processing
            processing_time = time.time() - start_time
            self.metrics.record_result(TaskResult.SUCCESS, processing_time)

            # Clean up processing tracking
            if task_id in self._processing_times:
                del self._processing_times[task_id]

            return TaskResult.SUCCESS

        except Exception as e:
            # Record error
            error_type = type(e).__name__
            processing_time = time.time() - start_time
            self.metrics.record_error(error_type)
            self.metrics.record_result(TaskResult.FAILURE, processing_time)

            # Log error
            if self.logger:
                self.logger.error(
                    f"Error processing task {task_id}: {str(e)}", exc_info=True
                )

            # Clean up processing tracking
            if task_id in self._processing_times:
                del self._processing_times[task_id]

            # Re-raise as conversation error
            context: Dict[str, Any] = {
                "task_id": task_id,
                "conversation_id": task.get("conversation_id"),
                "processing_time": processing_time,
            }
            raise ConversationProcessingError(
                f"Failed to process conversation task: {str(e)}",
                cause=e,
                context=context,
            )

    def _generate_response(
        self, conversation_id: int, message: str, context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate a response to a conversation message.

        Args:
            conversation_id: ID of the conversation
            message: The message to respond to
            context: Additional context for response generation

        Returns:
            Optional[str]: Generated response or None if generation fails
        """
        try:
            # Get conversation history if needed
            if context.get("use_history", True):
                conversation = self.conversation_manager.get_conversation(
                    conversation_id
                )
                messages = conversation["messages"]

                # Build conversation history string
                history = "\n".join(
                    [
                        f"{msg['speaker']}: {msg['text']}"
                        for msg in messages[-context.get("history_length", 5) :]
                    ]
                )

                # Cache conversation for later use
                self._conversation_cache[conversation_id] = {
                    "updated_at": time.time(),
                    "messages": messages,
                }
            else:
                history = f"user: {message}"

            # Use parser_refiner to generate a response
            # This is a simplified version - in a real implementation,
            # you would use a more sophisticated method using ParserRefiner
            # to leverage lexical knowledge

            # Simple response generation
            # (would be replaced by actual parser_refiner logic)
            # Use history to analyze the conversation context
            input_text = history if context.get("use_history", True) else message
            topic_words = self.parser_refiner.term_extractor.extract_terms(
                input_text, [], ""
            )[0]

            if topic_words:
                # Get the most relevant topic words
                topic = topic_words[0]

                # Try to use the parser to generate a meaningful response
                # based on lexical knowledge
                try:
                    # Process the topic word to ensure it's in the database
                    self.parser_refiner.process_word(topic)

                    # Get word entry from database for knowledge
                    word_entry = self.db_manager.get_word_entry(topic)

                    if word_entry and word_entry.get("definition"):
                        definition = word_entry["definition"]
                        response = f"About '{topic}': {definition}"
                    else:
                        response = f"I've noted your interest in '{topic}'. Could you tell me more?"
                except Exception:
                    # Fallback response if parsing fails
                    response = (
                        f"That's interesting! Can you tell me more about {topic}?"
                    )
            else:
                # Generic fallback
                response = "I'm processing that. Could you provide more details?"

            return response

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return "I'm sorry, I'm having trouble processing that right now."

    def _clear_stale_processing(self) -> None:
        """Clear any stale processing entries that have timed out."""
        current_time = time.time()
        stale_tasks: List[str] = []

        for task_id, start_time in self._processing_times.items():
            if current_time - start_time > self.processing_timeout:
                stale_tasks.append(task_id)

        for task_id in stale_tasks:
            if self.logger:
                self.logger.warning(f"Clearing stale task {task_id}")
            del self._processing_times[task_id]

    def get_status(self) -> ConversationWorkerStatus:
        """
        Get current worker status information.

        Returns:
            ConversationWorkerStatus: Current status of the worker
        """
        queue_size = self.queue_manager.size
        metrics_dict = self.metrics.get_metrics_dict()

        return self.state_tracker.to_dict(queue_size, metrics_dict)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current worker performance metrics.

        Returns:
            Dict[str, Any]: Dictionary containing performance metrics.
        """
        return self.metrics.get_metrics_dict()

    def process_task(self, task: ConversationTask) -> bool:
        """
        Process a conversation task directly (synchronously).

        Useful for immediate processing without queueing.

        Args:
            task: The conversation task to process

        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            result = self._process_task(task)
            return result == TaskResult.SUCCESS
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error in synchronous task processing: {str(e)}", exc_info=True
                )
            return False

    def submit_message(
        self, conversation_id: Optional[int], message: str, speaker: str
    ) -> Optional[str]:
        """
        Submit a message for processing and return a task ID.

        This is a convenience method that creates a task and enqueues it.

        Args:
            conversation_id: ID of the conversation, or None for a new conversation
            message: The message to process
            speaker: The speaker of the message

        Returns:
            Optional[str]: Task ID if successfully queued, None otherwise
        """
        try:
            task_id = f"task_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

            task: ConversationTask = {
                "task_id": task_id,
                "conversation_id": conversation_id,
                "message": message,
                "speaker": speaker,
                "timestamp": time.time(),
                "priority": 1,
                "context": {"generate_response": True},
            }

            # Enqueue the task
            self.queue_manager.enqueue(task)

            return task_id

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error submitting message: {str(e)}", exc_info=True)
            return None
