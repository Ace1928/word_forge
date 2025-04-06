import datetime
import logging
import os
import signal
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TypedDict, Union, cast

from word_forge.database.db_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager
from word_forge.graph.graph_manager import GraphManager
from word_forge.parser.parser_refiner import ParserRefiner
from word_forge.queue.queue_manager import EmptyQueueError, QueueManager


class WorkerState(Enum):
    """Worker thread lifecycle states for precise control and monitoring."""

    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


class WorkerEvent(TypedDict, total=False):
    """Structured event data for worker activity tracking."""

    timestamp: float
    event_type: str
    term: Optional[str]
    duration: Optional[float]
    error: Optional[str]
    details: Dict[str, Any]
    error_count: Optional[int]


@dataclass
class WorkerStatistics:
    """
    Runtime statistics for monitoring worker performance.

    Statistics are accumulated during operation and can be queried
    without interrupting the worker's processing loop.
    """

    start_time: float = 0.0
    processing_count: int = 0
    error_count: int = 0
    empty_queue_count: int = 0
    last_processed: Optional[str] = None
    last_error: Optional[str] = None
    last_active: float = 0.0
    processing_times: List[float] = field(default_factory=list)
    recent_events: List[WorkerEvent] = field(default_factory=list)
    max_events: int = 100

    @property
    def runtime_seconds(self) -> float:
        """Total runtime of the worker in seconds."""
        return time.time() - self.start_time

    @property
    def processing_rate(self) -> float:
        """Calculate average processing rate per minute."""
        runtime_minutes = self.runtime_seconds / 60
        if runtime_minutes <= 0:
            return 0.0
        return self.processing_count / runtime_minutes

    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time in seconds."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    @property
    def idle_seconds(self) -> float:
        """Seconds since the worker was last active."""
        if self.last_active == 0:
            return 0.0
        return time.time() - self.last_active

    def add_event(self, event: WorkerEvent) -> None:
        """Add an event to the recent events list, maintaining max size."""
        event["timestamp"] = event.get("timestamp", time.time())
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_events:
            self.recent_events.pop(0)
        self.last_active = time.time()

    def record_processing_time(self, duration: float) -> None:
        """Record a single processing time in seconds."""
        self.processing_times.append(duration)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)

    def reset(self) -> None:
        """Reset all statistics."""
        self.start_time = time.time()
        self.processing_count = 0
        self.error_count = 0
        self.empty_queue_count = 0
        self.last_processed = None
        self.last_error = None
        self.last_active = 0.0
        self.processing_times = []
        self.recent_events = []


class ProcessingResult(TypedDict, total=False):
    """
    Structured result of a word processing operation with enhanced metadata.

    Captures comprehensive processing context and results for analysis,
    knowledge graph construction, and adaptive learning.
    """

    success: bool
    term: str
    duration: float
    error: Optional[str]
    relationships_count: int
    new_terms_count: int
    relationship_types: Dict[str, int]  # Counts by relationship type
    processing_depth: int  # Depth level used during processing
    semantic_centrality: Optional[float]  # Measure of term centrality if known


# Type alias for worker result callback function
WorkerCallbackType = Callable[[ProcessingResult], None]


class WordForgeWorker(threading.Thread):
    """
    A robust background worker that processes queued words continuously.

    This worker implements:
    1. State management for clean lifecycle transitions
    2. Comprehensive statistics collection for monitoring
    3. Graceful error handling with automatic recovery
    4. Clean shutdown capability through signal handling
    """

    def __init__(
        self,
        parser_refiner: ParserRefiner,
        queue_manager: QueueManager[str],
        emotion_manager: EmotionManager,
        db_manager: Optional[DBManager] = None,
        embedder: Optional[Any] = "intfloat/multilingual-e5-large-v2",
        vector_store: Optional[Any] = "data/chroma.sqlite3",
        sleep_interval: float = 1.0,
        error_backoff_factor: float = 1.5,
        max_errors_per_minute: int = 10,
        daemon: bool = True,
        logger: Optional[logging.Logger] = None,
        result_callback: Optional[WorkerCallbackType] = None,
        launch_auxiliary_workers: bool = True,
    ):
        """
        Initialize the worker thread with required components.

        Args:
            parser_refiner: Component that processes and refines word data
            queue_manager: Queue system for word processing
            db_manager: Database interface for persistent storage
            sleep_interval: Time to wait when queue is empty
            error_backoff_factor: Multiplier for increasing wait time after errors
            max_errors_per_minute: Threshold before entering error backoff
            daemon: Whether thread should be a daemon (terminates with main program)
            logger: Logger for operational messages
            result_callback: Function to call with processing results
            launch_auxiliary_workers: Whether to start supporting worker threads
        """
        super().__init__(daemon=daemon)
        self.parser_refiner = parser_refiner
        self.queue_manager = queue_manager
        self.emotion_manager = emotion_manager
        self.db_manager = db_manager
        self.embedder = embedder
        self.vector_store = vector_store
        self.sleep_interval = sleep_interval
        self.error_backoff_factor = error_backoff_factor
        self.max_errors_per_minute = max_errors_per_minute
        self.result_callback = result_callback
        self.logger = logger or logging.getLogger(__name__)

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_until = 0.0
        self._last_error_time = 0.0
        self._error_count_window = []
        self._current_backoff = sleep_interval

        self._state = WorkerState.INITIALIZING
        self.stats = WorkerStatistics()
        self.stats.start_time = time.time()

        self.auxiliary_workers: List[threading.Thread] = []
        self.launch_auxiliary_workers = launch_auxiliary_workers

        self._register_signal_handlers()

        self._state_lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._recent_errors: List[float] = []
        self._processed_terms: set[str] = set()
        self._initial_queue_size = (
            queue_manager.size
            if callable(getattr(queue_manager, "size", None))
            else queue_manager.size
        )
        self._productivity_metric = 1.0
        self.base_sleep_interval = sleep_interval

        self._term_frequencies: Dict[str, int] = {}
        self._term_dependency_graph: Dict[str, List[str]] = {}
        self._consecutive_fast_processes: int = 0
        self._semantic_clusters: Dict[str, List[str]] = {}
        self._processing_patterns: List[str] = []

    def _register_signal_handlers(self) -> None:
        """Register handlers for graceful shutdown on system signals."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_shutdown_signal)

    def _handle_shutdown_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals by requesting graceful stop."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.request_stop()

    @property
    def state(self) -> WorkerState:
        """Get the current worker state."""
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, new_state: WorkerState) -> None:
        """Set the worker state with thread safety."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            self.logger.info(
                f"Worker state changed: {old_state.name} -> {new_state.name}"
            )

    def get_statistics(
        self,
    ) -> Dict[str, Union[int, float, str, None, List[Any], Dict[str, Any]]]:
        """
        Get worker statistics as a dictionary for monitoring.

        Returns:
            Dictionary with runtime statistics and state information
        """
        runtime = self.stats.runtime_seconds
        queue_size = self.queue_manager.size
        unique_words = len(list(self.queue_manager.iter_seen()))

        # Calculate productivity metric: processed words relative to queue growth
        if self._initial_queue_size > 0:
            words_processed = self.stats.processing_count
            queue_growth = max(0, queue_size - self._initial_queue_size)
            if words_processed > 0:
                self._productivity_metric = words_processed / (
                    words_processed + queue_growth
                )

        # Create a snapshot of current statistics
        stats: Dict[str, Union[int, float, str, None, List[Any], Dict[str, Any]]] = {
            "state": self.state.name,
            "runtime_seconds": runtime,
            "runtime_formatted": str(datetime.timedelta(seconds=int(runtime))),
            "processed_count": self.stats.processing_count,
            "error_count": self.stats.error_count,
            "empty_queue_count": self.stats.empty_queue_count,
            "processing_rate_per_minute": self.stats.processing_rate,
            "avg_processing_time": self.stats.average_processing_time,
            "queue_size": queue_size,
            "total_unique_words": unique_words,
            "last_processed": self.stats.last_processed,
            "last_error": self.stats.last_error,
            "idle_seconds": self.stats.idle_seconds,
            "productivity_metric": self._productivity_metric,
            "recent_events": self.stats.recent_events[-5:],  # Last 5 events
            "performance": {
                "mean_processing_time": self.stats.average_processing_time,
                "current_sleep_interval": self.sleep_interval,
                "error_rate_per_minute": len(self._recent_errors),
            },
        }

        return stats

    def formatted_statistics(self) -> str:
        """
        Get a formatted string representation of worker statistics.

        Returns:
            Human-readable statistics summary
        """
        stats = self.get_statistics()

        # Handle the avg_processing_time which might be of various types
        avg_time = stats["avg_processing_time"]
        if avg_time is None or not isinstance(avg_time, (int, float)):
            avg_time_ms = 0.0
        else:
            avg_time_ms = float(avg_time) * 1000

        return (
            f"Status: {stats['state']} | "
            f"Processed: {stats['processed_count']} | "
            f"Rate: {stats['processing_rate_per_minute']:.1f}/min | "
            f"Queue: {stats['queue_size']} | "
            f"Unique: {stats['total_unique_words']} | "
            f"Avg time: {avg_time_ms:.1f}ms"
        )

    def run(self) -> None:
        """Main worker thread execution loop."""
        self.logger.info("WordForgeWorker started")
        self.state = WorkerState.RUNNING

        if self.launch_auxiliary_workers and self.db_manager:
            self._start_auxiliary_workers()

        try:
            while not self._stop_requested.is_set():
                # Check if we're paused
                if self._pause_event.is_set():
                    if time.time() < self._pause_until:
                        time.sleep(0.1)
                        continue
                    else:
                        self._pause_event.clear()
                        self.state = WorkerState.RUNNING

                try:
                    # Process one item from the queue
                    self._process_next_word()
                except Exception as e:
                    self._handle_unexpected_error(e)
                    self._sleep_with_interruption(self._current_backoff)

        except Exception as e:
            self.state = WorkerState.ERROR
            self.logger.error(
                f"Fatal error in worker thread: {e}\n{traceback.format_exc()}"
            )
            self.stats.last_error = str(e)

        finally:
            self.state = WorkerState.STOPPING
            self._stop_auxiliary_workers()
            self.state = WorkerState.STOPPED
            self.logger.info("WordForgeWorker stopped")

    def stop(self) -> None:
        """Request the worker to stop gracefully."""
        self.logger.info("Stop requested")
        self._stop_requested.set()

    def _process_next_word(self) -> None:
        """Process the next word from the queue with recursive intelligence."""
        try:
            # Replace next_item() with dequeue()
            dequeue_result = self.queue_manager.dequeue(block=False)
            if dequeue_result.is_failure:
                raise EmptyQueueError()
            term = dequeue_result.unwrap()

            # Check if we've already processed this term
            if term in self._processed_terms:
                # Update term frequency metrics but don't reprocess
                self._update_term_frequency(term)
                return

            # Add to processed terms set
            self._processed_terms.add(term)

            # Calculate processing depth based on term and system state
            current_queue_depth = self.queue_manager.size if self.queue_manager else 0
            processing_depth = self._calculate_processing_depth(
                term, current_queue_depth
            )

            # Start timing
            start_time = time.time()

            # Process the word
            process_success = self.parser_refiner.process_word(term)

            # Calculate duration
            duration = time.time() - start_time

            # Create result dictionary from the boolean result
            result = {
                "success": process_success,
                "term": term,
                "duration": duration,
                "relationships_count": 0,  # Default values to be updated later
                "relationship_types": {},
                "new_terms_count": 0,
                "processing_depth": processing_depth,
            }

            # Get word entry if db_manager is available
            word_entry = None
            if self.db_manager is not None:
                try:
                    word_entry = self.db_manager.get_word_entry(term)
                except AttributeError:
                    self.logger.warning(
                        f"db_manager does not have get_word_entry method for term: {term}"
                    )

            # Gather relationship data if word exists in database
            relationship_counts: Dict[str, int] = {}
            if word_entry:
                # Safely extract relationships as a sequence of dictionaries
                relationships = word_entry.get("relationships", [])
                # Cast to appropriate type for compatibility
                relationships_compatible = cast(Sequence[Dict[str, Any]], relationships)
                relationship_counts = self._categorize_relationships(
                    relationships_compatible
                )
                self._update_term_frequency(term)
                self._processing_patterns.append(term)

                # Convert strings to dictionary format for _categorize_relationships
                recent_patterns = self._get_recent_term_pattern()
                self.logger.info(f"Recent term patterns: {recent_patterns}")

                # Create dictionary objects for each pattern string
                pattern_dicts: List[Dict[str, Any]] = [
                    {"term": p} for p in recent_patterns
                ]
                relationship_types = self._categorize_relationships(pattern_dicts)

                self._update_term_frequency(term)
                self._processing_patterns.append(term)
                self.logger.info(f"Processing patterns: {self._processing_patterns}")
                self.logger.info(f"Relationship types: {relationship_types}")

            # Create result object with properly typed fields
            result: Dict[str, Any] = {
                "success": True,
                "term": term,
                "duration": duration,
                "relationships_count": len(relationship_counts),
                "relationship_types": relationship_counts,
                "new_terms_count": 0,  # To be updated later
                "processing_depth": processing_depth,
            }

            # Handle successful processing
            self._handle_successful_processing(term, result, duration)

            # Update relationship tracking
            self._record_term_relationships(term, result.get("new_terms_count", 0))
            self._reorder_queue_if_needed(term, relationship_counts)

            # Invoke callback if provided
            if self.result_callback:
                self.result_callback(cast(ProcessingResult, result))

        except EmptyQueueError:
            self._handle_empty_queue()

    def _calculate_processing_depth(self, term: str, queue_size: int) -> int:
        """
        Determine optimal processing depth based on term significance and system load.

        Returns:
            Processing depth level (1-5) with higher values for deeper analysis
        """
        # Base depth starts at 3 (medium)
        depth = 3

        # Increase depth for shorter, potentially more fundamental terms
        if len(term) <= 5:
            depth += 1

        # Reduce depth when queue is very large to prevent resource exhaustion
        if queue_size > 1000:
            depth -= 1

        # Adjust based on recent processing performance
        if self.stats.average_processing_time > 5.0:
            depth -= 1

        # Ensure depth stays within valid range (1-5)
        return max(1, min(5, depth))

    def _update_term_frequency(self, term: str) -> None:
        """Track how often terms are re-encountered to identify core concepts."""
        if term not in self._term_frequencies:
            self._term_frequencies[term] = 0

        self._term_frequencies[term] += 1

        # If a term appears frequently, it might be semantically central
        if self._term_frequencies[term] >= 5:

            self.logger.info(f"Term '{term}' is frequently encountered.")

    def _get_recent_term_pattern(self) -> List[str]:
        """Identify recent processing patterns to detect semantic clusters."""
        patterns: List[str] = []
        for event in self.stats.recent_events[-10:]:
            if event.get("event_type") == "processed" and event.get("term") is not None:
                term = event.get("term", "")
                if isinstance(term, str):
                    patterns.append(term)
        return patterns

    def _categorize_relationships(
        self, relationships: Sequence[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Categorize relationship types for semantic analysis."""
        type_counts: Dict[str, int] = {}
        for rel in relationships:
            # Handle both regular relationship objects and our pattern objects
            rel_type = rel.get("type", "unknown")
            if rel_type not in type_counts:
                type_counts[rel_type] = 0
            type_counts[rel_type] += 1
        return type_counts

    def _handle_successful_processing(
        self, term: str, result: Dict[str, Any], duration: float
    ) -> None:
        """Process and record successful term processing."""
        self.stats.processing_count += 1
        self.stats.last_processed = term
        self.stats.record_processing_time(duration)

        # Reset sleep interval after successful processing
        self.sleep_interval = self.base_sleep_interval

        # Record event with comprehensive metadata
        self.stats.add_event(
            {
                "timestamp": time.time(),
                "event_type": "processed",
                "term": term,
                "duration": duration,
                "details": {
                    "relationships": result.get("relationships_count", 0),
                    "new_terms": result.get("new_terms_count", 0),
                    "relationship_types": result.get("relationship_types", {}),
                    "processing_depth": result.get("processing_depth", 3),
                },
            }
        )

        # Adaptive learning: if processing was very fast, process more items before sleeping
        if duration < 0.1:
            self._consecutive_fast_processes += 1
        else:
            self._consecutive_fast_processes = 0

    def _handle_failed_processing(self, term: str, result: Dict[str, Any]) -> None:
        """Handle and record failed term processing."""
        self.stats.error_count += 1
        self.stats.add_event(
            {
                "timestamp": time.time(),
                "event_type": "processing_failed",
                "term": term,
                "error": result.get("error", "Unknown error"),
                "details": {"duration": result.get("duration", 0)},
            }
        )
        self.logger.warning(
            f"Failed to process term '{term}': {result.get('error', 'Unknown error')}"
        )

    def _record_term_relationships(self, term: str, new_terms_count: int) -> None:
        """Track term relationships for dependency analysis."""
        if term not in self._term_dependency_graph:
            self._term_dependency_graph[term] = []

        if new_terms_count > 0:
            # This term generated new terms for the queue
            self.stats.add_event(
                {
                    "timestamp": time.time(),
                    "event_type": "term_relationships",
                    "term": term,
                    "details": {"new_terms": new_terms_count},
                }
            )

    def _reorder_queue_if_needed(
        self, term: str, relationship_types: Dict[str, int]
    ) -> None:
        """
        Intelligently reorder the queue based on processing insights.
        For example, prioritize terms with strong synonym relationships.
        """
        # If this is a rich semantic term (many synonyms), process related terms next
        if relationship_types.get("synonym", 0) > 5 and hasattr(
            self.queue_manager, "prioritize"
        ):
            self.logger.debug(
                f"Prioritizing terms related to '{term}' due to high synonym count"
            )
            # Extract related terms and prioritize them
            related_terms = self._get_related_terms(term)
            for related_term in related_terms:
                # Use getattr to safely call the prioritize method
                prioritize_method = getattr(self.queue_manager, "prioritize", None)
                if callable(prioritize_method):
                    prioritize_method(related_term)
                else:
                    self.logger.debug(
                        f"Cannot prioritize term '{related_term}': QueueManager has no prioritize method"
                    )

    def _get_related_terms(self, term: str) -> List[str]:
        """Get terms related to the given term from our dependency graph."""
        return self._term_dependency_graph.get(term, [])

    def _handle_empty_queue(self) -> None:
        """Handle empty queue with adaptive sleep duration."""
        self.stats.empty_queue_count += 1

        # Record empty queue encounter only occasionally to avoid spam
        if self.stats.empty_queue_count % 10 == 1:
            self.logger.debug("Queue is empty, waiting for new items")
            self.stats.add_event({"event_type": "empty_queue"})

        # Adaptive sleep based on processing history
        if self.stats.processing_count > 0:
            # If we've processed many terms, take a longer break
            sleep_time = min(self.sleep_interval * 2, 5.0)
        else:
            # Otherwise use normal interval
            sleep_time = self.sleep_interval

        self._sleep_with_interruption(sleep_time)

    def _handle_unexpected_error(self, error: Exception) -> None:
        """
        Handle unexpected errors during processing with escalating responses.

        Args:
            error: The exception that was caught
        """
        self.stats.error_count += 1
        self.stats.last_error = str(error)
        error_trace = traceback.format_exc()
        self.logger.error(f"Unexpected error: {error}\n{error_trace}")

        # Record error event
        self.stats.add_event(
            {
                "timestamp": time.time(),
                "event_type": "error",
                "error": str(error),
                "details": {"traceback": error_trace},
            }
        )

        # Track error rate for circuit breaking
        current_time = time.time()
        self._recent_errors.append(current_time)

        # Remove errors older than 1 minute from tracking
        self._recent_errors = [t for t in self._recent_errors if current_time - t < 60]

        # Check error frequency and trigger backoff if needed
        if len(self._recent_errors) >= self.max_errors_per_minute:
            self._trigger_error_backoff()

    def _trigger_error_backoff(self) -> None:
        """Apply exponential backoff when error rate is too high."""
        self._current_backoff = min(
            self._current_backoff * self.error_backoff_factor, 30.0
        )
        self.logger.warning(
            f"Error threshold exceeded. Backing off for {self._current_backoff:.2f} seconds"
        )
        self.stats.add_event(
            {
                "event_type": "backoff",
                "duration": self._current_backoff,
                "error_count": len(self._recent_errors),
            }
        )

    def _sleep_with_interruption(self, seconds: float) -> None:
        """Sleep for the specified duration, but stop early if requested."""
        end_time = time.time() + seconds
        while time.time() < end_time:
            if self._stop_requested.is_set():
                break
            time.sleep(0.1)

    def request_stop(self) -> None:
        """Request the worker to stop gracefully."""
        self.logger.info("Stop requested")
        self._stop_requested.set()

    def pause(self, seconds: Optional[float] = None) -> None:
        """
        Pause the worker for a specified duration.

        Args:
            seconds: Duration to pause for, or indefinitely if None
        """
        self.state = WorkerState.PAUSED
        self._pause_event.set()
        if seconds is not None:
            self._pause_until = time.time() + seconds
            self.logger.info(f"Worker paused for {seconds:.1f} seconds")
        else:
            self._pause_until = float("inf")
            self.logger.info("Worker paused indefinitely")

    def resume(self) -> None:
        """Resume worker from a paused state."""
        if self._pause_event.is_set():
            self._pause_event.clear()
            self._pause_until = 0
            self.state = WorkerState.RUNNING
            self.logger.info("Worker resumed")

    def _start_auxiliary_workers(self) -> None:
        """Start auxiliary workers to handle specialized tasks."""
        self.logger.info("Starting auxiliary workers")
        launched_workers: List[threading.Thread] = []

        # Try to start each type of worker
        self._try_start_graph_worker(launched_workers)
        self._try_start_emotion_worker(launched_workers)
        self._try_start_vector_worker(launched_workers)

        if len(launched_workers) > 0:
            self.logger.info(f"Started {len(launched_workers)} auxiliary workers")
        else:
            self.logger.info("No auxiliary workers were started")

    def _try_start_graph_worker(self, launched_workers: List[threading.Thread]) -> None:
        """Try to start a graph analysis worker if dependencies are available."""
        try:
            # Import only if we need it to avoid unnecessary dependencies
            from word_forge.graph.graph_worker import GraphWorker

            # Only create GraphManager if db_manager is not None
            if self.db_manager is None:
                self.logger.debug("Cannot start graph worker: db_manager is None")
                return

            graph_worker = GraphWorker(
                graph_manager=GraphManager(db_manager=self.db_manager)
            )
            graph_worker.daemon = True
            graph_worker.start()

            self.auxiliary_workers.append(graph_worker)
            launched_workers.append(graph_worker)
            self.logger.info("Started graph analysis worker")
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"Could not start graph worker: {e}")

    def _try_start_emotion_worker(
        self, launched_workers: List[threading.Thread]
    ) -> None:
        """Try to start an emotion analysis worker if dependencies are available."""
        try:
            # Import only if we need it to avoid unnecessary dependencies
            from word_forge.emotion.emotion_worker import EmotionWorker

            # Ensure db_manager is not None before starting worker
            if self.db_manager is None:
                self.logger.debug("Cannot start emotion worker: db_manager is None")
                return

            # We've already checked db_manager is not None, so we can safely assert this for type checking
            assert (
                self.db_manager is not None
            ), "db_manager should not be None at this point"

            emotion_worker = EmotionWorker(
                db=self.db_manager,
                emotion_manager=self.emotion_manager,
                poll_interval=5.0,
                batch_size=10,
            )
            emotion_worker.daemon = True
            emotion_worker.start()

            self.auxiliary_workers.append(emotion_worker)
            launched_workers.append(emotion_worker)
            self.logger.info("Started emotion analysis worker")
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"Could not start emotion worker: {e}")

    def _try_start_vector_worker(
        self, launched_workers: List[threading.Thread]
    ) -> None:
        """Try to start a vector embedding worker if dependencies are available."""
        try:
            # Import only if we need it
            from word_forge.vectorizer.vector_worker import VectorWorker

            # Ensure db_manager and vector_store are not None before starting worker
            if self.db_manager is None:
                self.logger.debug("Cannot start vector worker: db_manager is None")
                return

            if self.vector_store is None:
                self.logger.debug("Cannot start vector worker: vector_store is None")
                return

            if self.embedder is None:
                self.logger.debug("Cannot start vector worker: embedder is None")
                return

            # Check if embedder is compatible with the Embedder protocol
            if not self.embedder:
                self.logger.debug("Cannot start vector worker: embedder is None")
                return

            vector_worker = VectorWorker(
                db=self.db_manager,
                vector_store=self.vector_store,
                embedder=self.embedder,  # Now we've verified this is either an Embedder or a string model name
                poll_interval=5.0,
                daemon=True,
                logger=self.logger,
            )
            vector_worker.start()

            self.auxiliary_workers.append(vector_worker)
            launched_workers.append(vector_worker)
            self.logger.info("Started vector embedding worker")
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"Could not start vector worker: {e}")

    def _stop_auxiliary_workers(self) -> None:
        """Stop all auxiliary workers gracefully."""
        if not self.auxiliary_workers:
            return

        self.logger.info(f"Stopping {len(self.auxiliary_workers)} auxiliary workers")
        remaining_workers: List[threading.Thread] = []

        # First try to stop workers gracefully
        for worker in self.auxiliary_workers:
            # Check if worker has stop method
            if hasattr(worker, "stop") and callable(getattr(worker, "stop", None)):
                try:
                    getattr(worker, "stop")()
                    self.logger.debug(f"Requested stop for {worker.__class__.__name__}")
                except Exception as e:
                    self.logger.debug(
                        f"Error stopping {worker.__class__.__name__}: {e}"
                    )
                    remaining_workers.append(worker)
            else:
                self.logger.debug(f"No stop method for {worker.__class__.__name__}")
                # Add to remaining so we can join it
                remaining_workers.append(worker)

        # Join all workers to ensure they've stopped
        for worker in self.auxiliary_workers:
            if worker.is_alive():
                worker.join(timeout=2.0)
                if worker.is_alive():
                    self.logger.warning(
                        f"Worker {worker.__class__.__name__} did not stop in time"
                    )
                    remaining_workers.append(worker)
                else:
                    self.logger.debug(
                        f"Worker {worker.__class__.__name__} stopped successfully"
                    )
            else:
                self.logger.debug(
                    f"Worker {worker.__class__.__name__} was already stopped"
                )

        self.auxiliary_workers = remaining_workers
        if len(self.auxiliary_workers) > 0:
            self.logger.warning(
                f"{len(self.auxiliary_workers)} workers could not be stopped gracefully"
            )


def create_progress_bar(current: int, total: int, width: int = 40) -> str:
    """
    Create a text-based progress bar.

    Args:
        current: Current progress value
        total: Maximum progress value
        width: Width of the progress bar in characters

    Returns:
        Formatted progress bar string
    """
    percent = current / total
    filled_width = int(width * percent)
    bar = "█" * filled_width + "░" * (width - filled_width)
    return f"[{bar}] {percent:.1%}"


def print_table(
    headers: List[str], rows: List[List[str]], title: Optional[str] = None
) -> None:
    """
    Print a formatted table to the console.

    Args:
        headers: List of column headers
        rows: List of rows, each a list of values aligned with headers
        title: Optional title for the table
    """
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Create separator line
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    # Print title if provided
    if title:
        title_width = len(separator) - 4
        print(f"┌{'─' * title_width}┐")
        print(f"│ {title.center(title_width)} │")
        print(f"└{'─' * title_width}┘")

    # Print headers
    print(separator)
    header_str = "|"
    for i, header in enumerate(headers):
        header_str += f" {header.center(col_widths[i])} |"
    print(header_str)
    print(separator)

    # Print rows
    for row in rows:
        row_str = "|"
        for i, cell in enumerate(row):
            row_str += f" {str(cell).ljust(col_widths[i])} |"
        print(row_str)

    print(separator)


def main() -> None:
    """
    Demonstrate the WordForgeWorker with a time-limited run showing detailed metrics.

    This function:
    1. Sets up logging and required components
    2. Populates an initial queue with seed words
    3. Creates and runs a worker for a fixed duration
    4. Displays rich statistics about the worker's performance
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("WordForgeDemo")

    # Create temporary database
    logger.info("Initializing temporary database and resources...")
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "word_forge_demo.sqlite")
    data_dir = Path(os.path.join(temp_dir, "data"))
    data_dir.mkdir(exist_ok=True)

    # Initialize components
    from word_forge.database.db_manager import DBManager
    from word_forge.parser.parser_refiner import ParserRefiner
    from word_forge.queue.queue_manager import QueueManager

    db_manager = DBManager(db_path)
    queue_manager = QueueManager[str]()
    parser_refiner = ParserRefiner(
        db_manager,
        queue_manager,
        data_dir=str(data_dir),
    )

    # Seed the queue with initial words
    seed_words = [
        "algorithm",
        "language",
        "computer",
        "data",
        "program",
        "network",
        "code",
        "function",
        "logic",
        "system",
    ]

    processed_results: List[ProcessingResult] = []

    # Process results callback
    def on_word_processed(result: ProcessingResult) -> None:
        """Callback that receives processing results."""
        processed_results.append(result)
        if result.get("success", False):
            logger.info(
                f"Processed '{result.get('term', '')}' in {result.get('duration', 0.0):.3f}s"
            )
        else:
            logger.warning(
                f"Failed to process '{result.get('term', '')}': {result.get('error', 'Unknown error')}"
            )

    logger.info(f"Seeding queue with {len(seed_words)} initial words...")
    for word in seed_words:
        queue_manager.enqueue(word)

    # Create and start worker
    logger.info("Creating worker thread...")
    worker = WordForgeWorker(
        parser_refiner,
        queue_manager,
        db_manager=db_manager,
        emotion_manager=EmotionManager(db_manager),
        sleep_interval=0.05,
        logger=logger,
        result_callback=on_word_processed,
    )

    # Run for specified time
    run_minutes = 1.0  # Demo runs for 1 minute
    run_seconds = run_minutes * 60

    logger.info(f"Starting worker for {run_minutes} minute(s)...")
    worker.start()

    # Monitor progress during runtime with periodic status updates
    update_interval = 5  # Show stats every 5 seconds
    start_time = time.time()
    end_time = start_time + run_seconds

    # Display processing milestones
    milestones = [10, 25, 50, 100, 250, 500, 1000]
    next_milestone_idx = 0

    try:
        while time.time() < end_time and worker.is_alive():
            # Sleep for update interval or until end time, whichever comes first
            sleep_time = min(update_interval, end_time - time.time())
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Print current stats
            stats = worker.get_statistics()
            processed = stats["processed_count"]

            # Print status update
            elapsed = time.time() - start_time
            remaining = run_seconds - elapsed
            logger.info(
                f"Status: {stats['state']} | "
                f"Processed: {processed} | "
                f"Rate: {stats['processing_rate_per_minute']:.1f}/min | "
                f"Queue: {stats['queue_size']} | "
                f"Time remaining: {int(remaining)}s"
            )

            # Check for processing milestones
            if (
                next_milestone_idx < len(milestones)
                and isinstance(processed, (int, float))
                and processed >= milestones[next_milestone_idx]
            ):
                milestone = milestones[next_milestone_idx]
                logger.info(f"🏆 Milestone reached: {milestone} words processed!")
                next_milestone_idx += 1

    except KeyboardInterrupt:
        logger.info("Demo interrupted. Stopping worker...")
        worker.request_stop()

    finally:
        # Request worker to stop if still running
        if worker.is_alive():
            worker.request_stop()
            worker.join(timeout=5.0)

        # Display final statistics in a table
        logger.info("\n📊 Final Processing Statistics:")
        stats = worker.get_statistics()

        headers = ["Metric", "Value"]
        rows = [
            ["Total Runtime", f"{stats['runtime_formatted']}"],
            ["Words Processed", f"{stats['processed_count']}"],
            ["Processing Rate", f"{stats['processing_rate_per_minute']:.1f} words/min"],
            [
                "Avg Processing Time",
                f"{(float(stats['avg_processing_time']) if isinstance(stats['avg_processing_time'], (int, float)) else 0.0)*1000:.1f} ms",
            ],
            ["Error Count", f"{stats['error_count']}"],
            ["Queue Items Remaining", f"{stats['queue_size']}"],
            ["Unique Words Seen", f"{stats['total_unique_words']}"],
        ]

        print("\n")
        print_table(headers, rows, "WordForge Processing Statistics")
        print("\n")

        # Cleanup temporary directory
        try:
            import shutil

            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {e}")


if __name__ == "__main__":
    main()
