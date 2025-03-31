import datetime
import json
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
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

from word_forge.database.db_manager import DBManager
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
        """Calculate total runtime in seconds since worker started."""
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    @property
    def idle_seconds(self) -> float:
        """Calculate idle time since last activity."""
        if self.last_active == 0:
            return 0.0
        return time.time() - self.last_active

    @property
    def processing_rate(self) -> float:
        """Calculate average processing rate (items per minute)."""
        runtime = self.runtime_seconds
        if runtime < 1:
            return 0.0
        return (self.processing_count / runtime) * 60

    @property
    def average_processing_time(self) -> float:
        """Calculate average time to process a word in seconds."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)

    def add_event(self, event: WorkerEvent) -> None:
        """Add an event to the recent events list with size limiting."""
        self.recent_events.append(event)
        # Keep only the most recent events
        if len(self.recent_events) > self.max_events:
            self.recent_events = self.recent_events[-self.max_events :]

    def record_processing_time(self, duration: float) -> None:
        """Record the time taken to process a word."""
        self.processing_times.append(duration)
        # Keep only recent processing times to avoid unbounded growth
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]

    def reset(self) -> None:
        """Reset all counters except start_time."""
        self.processing_count = 0
        self.error_count = 0
        self.empty_queue_count = 0
        self.last_processed = None
        self.last_error = None
        self.processing_times.clear()
        self.recent_events.clear()


class ProcessingResult(TypedDict):
    """Structured result of a word processing operation."""

    success: bool
    term: str
    duration: float
    error: Optional[str]
    relationships_count: int
    new_terms_count: int


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
        queue_manager: QueueManager,
        db_manager: Optional[DBManager] = None,
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
        self.db_manager = db_manager
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

        self.auxiliary_workers = []
        self.launch_auxiliary_workers = launch_auxiliary_workers

        self._register_signal_handlers()

        self._state_lock = threading.RLock()
        self._stop_requested = threading.Event()
        self._recent_errors = []
        self._processed_terms = set()
        self._initial_queue_size = queue_manager.size()
        self._productivity_metric = 1.0
        self.base_sleep_interval = sleep_interval

    def _register_signal_handlers(self) -> None:
        """Register handlers for graceful shutdown on system signals."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle_shutdown_signal)

    def _handle_shutdown_signal(self, signum: int, frame) -> None:
        """Handle shutdown signals by requesting graceful stop."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.request_stop()

    @property
    def state(self) -> WorkerState:
        """Get the current worker state safely across threads."""
        with self._state_lock:
            return self._state

    @state.setter
    def state(self, new_state: WorkerState) -> None:
        """Update worker state with proper locking and logging."""
        with self._state_lock:
            old_state = self._state
            self._state = new_state

        if old_state != new_state:
            self.logger.info(
                f"Worker state changed: {old_state.name} → {new_state.name}"
            )

            # Record state change as an event
            self.stats.add_event(
                {
                    "timestamp": time.time(),
                    "event_type": "state_change",
                    "details": {
                        "old_state": old_state.name,
                        "new_state": new_state.name,
                    },
                }
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
        queue_size = self.queue_manager.size()
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
        stats = {
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

        return (
            f"Status: {stats['state']} | "
            f"Processed: {stats['processed_count']} | "
            f"Rate: {stats['processing_rate_per_minute']:.1f}/min | "
            f"Queue: {stats['queue_size']} | "
            f"Unique: {stats['total_unique_words']} | "
            f"Avg time: {stats['avg_processing_time']*1000:.1f}ms"
        )

    def run(self) -> None:
        """Main worker thread execution loop."""
        self.logger.info("WordForgeWorker started")
        self.state = WorkerState.RUNNING

        if self.launch_auxiliary_workers and self.db_manager:
            self._start_auxiliary_workers()

        try:
            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    if time.time() >= self._pause_until or self._pause_until == 0:
                        self.resume()
                    else:
                        time.sleep(0.1)
                        continue

                try:
                    self._process_next_word()
                except EmptyQueueError:
                    self.stats.empty_queue_count += 1
                    self._sleep_with_interruption(self.sleep_interval)
                except Exception as e:
                    self._handle_unexpected_error(e)

        except Exception as e:
            self.state = WorkerState.ERROR
            self.logger.error(f"Critical error in worker thread: {str(e)}")
            import traceback

            self.logger.debug(traceback.format_exc())

        finally:
            self._stop_auxiliary_workers()
            self.state = WorkerState.STOPPED
            self.logger.info("WordForgeWorker stopped")

    def _process_next_word(self) -> None:
        """
        Process the next word from the queue, handling empty queue condition.

        This is the core processing logic that dequeues and processes words,
        updates statistics, and manages sleep intervals.
        """
        try:
            # Attempt to get a word from the queue
            term = self.queue_manager.dequeue()

            # Skip if we've already processed this term
            if term in self._processed_terms:
                return

            self._processed_terms.add(term)

            # We have a word to process
            start_time = time.time()
            self.stats.last_active = start_time
            self.logger.debug(f"Processing: '{term}'")

            # Process the word and track result
            initial_queue_size = self.queue_manager.size()
            success = self.parser_refiner.process_word(term)
            end_time = time.time()
            processing_duration = end_time - start_time

            # Get relationships count from database
            relationships_count = 0
            new_terms_count = self.queue_manager.size() - initial_queue_size

            if self.db_manager:
                word_entry = self.db_manager.get_word_if_exists(term)
                if word_entry:
                    relationships_count = len(word_entry.get("relationships", []))

            # Record result
            result: ProcessingResult = {
                "success": success,
                "term": term,
                "duration": processing_duration,
                "error": None,
                "relationships_count": relationships_count,
                "new_terms_count": new_terms_count,
            }

            if success:
                self.stats.processing_count += 1
                self.stats.last_processed = term
                self.stats.record_processing_time(processing_duration)
                # Reset sleep interval after successful processing
                self.sleep_interval = self.base_sleep_interval

                # Record successful processing event
                self.stats.add_event(
                    {
                        "timestamp": time.time(),
                        "event_type": "processed",
                        "term": term,
                        "duration": processing_duration,
                        "details": {
                            "relationships": relationships_count,
                            "new_terms": new_terms_count,
                        },
                    }
                )
            else:
                # Record error
                self.stats.error_count += 1
                result["error"] = f"Processing failed for term: {term}"
                self.stats.add_event(
                    {
                        "timestamp": time.time(),
                        "event_type": "processing_failed",
                        "term": term,
                        "error": result["error"],
                    }
                )

            # Invoke callback if provided
            if self.result_callback:
                try:
                    self.result_callback(result)
                except Exception as callback_error:
                    self.logger.error(f"Error in result callback: {callback_error}")

        except EmptyQueueError:
            # No items in queue, sleep and track empty queue event
            self.stats.empty_queue_count += 1

            # Record empty queue encounter only occasionally to avoid spam
            if self.stats.empty_queue_count % 10 == 1:
                self.stats.add_event(
                    {
                        "timestamp": time.time(),
                        "event_type": "empty_queue",
                    }
                )

            self._sleep_with_interruption(self.sleep_interval)

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

        # Remove errors older than 60 seconds
        self._recent_errors = [t for t in self._recent_errors if current_time - t < 60]

        # Implement circuit breaking if error rate exceeds threshold
        if len(self._recent_errors) >= self.max_errors_per_minute:
            self._trigger_error_backoff()
        else:
            # Increase sleep time for gradual backoff
            self.sleep_interval *= self.error_backoff_factor
            self.logger.warning(
                f"Increased sleep interval to {self.sleep_interval:.2f}s after error"
            )
            self._sleep_with_interruption(self.sleep_interval)

    def _trigger_error_backoff(self) -> None:
        """
        Trigger a circuit-breaker pause after excessive errors.

        Pauses the worker for a period to allow systems to recover.
        """
        pause_minutes = min(15, len(self._recent_errors) // self.max_errors_per_minute)
        self._pause_until = time.time() + (pause_minutes * 60)

        self.logger.warning(
            f"Too many errors ({len(self._recent_errors)} in the last minute). "
            f"Pausing worker for {pause_minutes} minutes."
        )

        # Record circuit breaker event
        self.stats.add_event(
            {
                "timestamp": time.time(),
                "event_type": "circuit_breaker",
                "details": {
                    "error_count": len(self._recent_errors),
                    "pause_minutes": pause_minutes,
                },
            }
        )

        self.state = WorkerState.PAUSED
        self._recent_errors.clear()

    def _sleep_with_interruption(self, seconds: float) -> None:
        """
        Sleep for specified duration but exit early if stop requested.

        Args:
            seconds: Time to sleep in seconds
        """
        self._stop_requested.wait(timeout=seconds)

    def request_stop(self) -> None:
        """
        Request a graceful worker shutdown.

        The worker will finish current processing and exit its run loop.
        """
        self.logger.info("Stop requested")
        self.state = WorkerState.STOPPING
        self._stop_requested.set()

    def pause(self, seconds: Optional[float] = None) -> None:
        """
        Pause the worker temporarily or indefinitely.

        Args:
            seconds: Duration to pause in seconds, or None for indefinite pause
        """
        if seconds is not None:
            self._pause_until = time.time() + seconds
            self.logger.info(f"Pausing worker for {seconds:.1f} seconds")
        else:
            # Use a very far future timestamp for indefinite pause
            self._pause_until = time.time() + (365 * 24 * 60 * 60)  # ~1 year
            self.logger.info("Pausing worker indefinitely")

        self.state = WorkerState.PAUSED

    def resume(self) -> None:
        """Resume a paused worker."""
        if self.state == WorkerState.PAUSED:
            self.logger.info("Resuming worker")
            self.state = WorkerState.RUNNING
            self.sleep_interval = self.base_sleep_interval

    def _start_auxiliary_workers(self) -> None:
        """Start additional specialized worker threads."""
        try:
            # Import here to avoid circular imports
            from word_forge.emotion.emotion_worker import EmotionManager, EmotionWorker
            from word_forge.graph.graph_worker import GraphManager, GraphWorker
            from word_forge.vectorizer.vector_worker import (
                StorageType,
                TransformerEmbedder,
                VectorStore,
                VectorWorker,
            )

            # Start Graph Worker
            graph_manager = GraphManager(self.db_manager)
            graph_worker = GraphWorker(
                graph_manager=graph_manager,
                poll_interval=60.0,  # Check every minute
                daemon=True,
            )
            graph_worker.start()
            self.auxiliary_workers.append(graph_worker)
            self.logger.info("Started auxiliary Graph Worker")

            # Start Emotion Worker
            emotion_manager = EmotionManager(self.db_manager)
            emotion_worker = EmotionWorker(
                db=self.db_manager,
                emotion_manager=emotion_manager,
                poll_interval=30.0,
                batch_size=20,
                daemon=True,
            )
            emotion_worker.start()
            self.auxiliary_workers.append(emotion_worker)
            self.logger.info("Started auxiliary Emotion Worker")

            # Start Vector Worker
            vector_store = VectorStore(
                storage_type=StorageType.DISK,
                dimension=int(1024),
            )
            embedder = TransformerEmbedder()
            vector_worker = VectorWorker(
                db=self.db_manager,
                vector_store=vector_store,
                embedder=embedder,
                poll_interval=45.0,
                daemon=True,
                logger=self.logger,
            )
            vector_worker.start()
            self.auxiliary_workers.append(vector_worker)
            self.logger.info("Started auxiliary Vector Worker")

        except ImportError as e:
            self.logger.warning(f"Could not start auxiliary workers: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error starting auxiliary workers: {str(e)}")

    def _stop_auxiliary_workers(self) -> None:
        """Stop all auxiliary worker threads gracefully."""
        for worker in self.auxiliary_workers:
            try:
                if hasattr(worker, "stop") and callable(worker.stop):
                    worker.stop()
                elif hasattr(worker, "request_stop") and callable(worker.request_stop):
                    worker.request_stop()
            except Exception as e:
                self.logger.warning(f"Error stopping auxiliary worker: {str(e)}")

        # Wait for workers to stop
        for worker in self.auxiliary_workers:
            try:
                worker.join(timeout=5.0)
            except Exception:
                pass

        self.logger.info(f"Stopped {len(self.auxiliary_workers)} auxiliary workers")
        self.auxiliary_workers = []


def create_demo_files(data_dir: Path) -> None:
    """
    Create sample dictionary files for demonstration.

    Args:
        data_dir: Directory to create files in
    """
    # Create demo dictionary as JSON
    demo_dict = {
        "algorithm": {
            "definition": "A step-by-step procedure for solving a problem",
            "part_of_speech": "noun",
            "examples": ["The sorting algorithm arranges elements in order."],
        },
        "data": {
            "definition": "Facts or information used for analysis or reasoning",
            "part_of_speech": "noun",
            "examples": ["The program processes large amounts of data."],
        },
    }

    # Create sample thesaurus entries
    thesaurus_entries = [
        {"word": "algorithm", "synonyms": ["procedure", "process", "method"]},
        {"word": "data", "synonyms": ["information", "facts", "figures"]},
    ]

    # Write sample files
    with open(data_dir / "openthesaurus.jsonl", "w") as f:
        for entry in thesaurus_entries:
            f.write(json.dumps(entry) + "\n")

    with open(data_dir / "odict.json", "w") as f:
        json.dump(demo_dict, f, indent=2)

    with open(data_dir / "opendict.json", "w") as f:
        json.dump(demo_dict, f, indent=2)

    with open(data_dir / "thesaurus.jsonl", "w") as f:
        for entry in thesaurus_entries:
            f.write(json.dumps(entry) + "\n")


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
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(cell)))

    # Create separator line
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    # Print title if provided
    if title:
        print(f"\n{title}")

    # Print headers
    print(separator)
    header_str = "|"
    for i, header in enumerate(headers):
        header_str += f" {header.ljust(col_widths[i])} |"
    print(header_str)
    print(separator)

    # Print rows
    for row in rows:
        row_str = "|"
        for i, cell in enumerate(row):
            if i < len(col_widths):
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

    # Create sample dictionary files
    create_demo_files(data_dir)

    # Initialize components
    from word_forge.parser.parser_refiner import ParserRefiner
    from word_forge.queue.queue_manager import QueueManager

    db_manager = DBManager(db_path)
    queue_manager = QueueManager[str]()
    parser_refiner = ParserRefiner(
        db_manager,
        queue_manager,
        data_dir=str(data_dir),
        enable_model=False,
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

    # Process results callback
    def on_word_processed(result: ProcessingResult) -> None:
        if result["success"]:
            logger.info(
                f"Processed '{result['term']}' in {result['duration']*1000:.1f}ms "
                f"(found {result['relationships_count']} relationships, "
                f"discovered {result['new_terms_count']} new terms)"
            )
        else:
            logger.warning(f"Failed to process '{result['term']}': {result['error']}")

    logger.info(f"Seeding queue with {len(seed_words)} initial words...")
    for word in seed_words:
        queue_manager.enqueue(word)

    # Create and start worker
    logger.info("Creating worker thread...")
    worker = WordForgeWorker(
        parser_refiner,
        queue_manager,
        db_manager=db_manager,
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
            # Sleep for a bit, but wake up if we reach end time
            remaining = min(update_interval, end_time - time.time())
            if remaining <= 0:
                break
            time.sleep(remaining)

            # Display current statistics
            stats = worker.get_statistics()
            logger.info(worker.formatted_statistics())

            # Check for milestone achievement
            if (
                next_milestone_idx < len(milestones)
                and stats["processed_count"] >= milestones[next_milestone_idx]
            ):
                logger.info(
                    f"🎉 Milestone: {milestones[next_milestone_idx]} words processed!"
                )
                next_milestone_idx += 1

            # Show progress bar
            elapsed = time.time() - start_time
            logger.info(
                f"Progress: {create_progress_bar(int(elapsed), int(run_seconds))}"
            )

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Stopping worker...")
        worker.request_stop()

    finally:
        # Request stop if still running
        if worker.is_alive():
            logger.info("Time's up. Stopping worker...")
            worker.request_stop()
            worker.join(timeout=5.0)

            # Display final statistics
            stats = worker.get_statistics()
            logger.info("\n=== Final Statistics ===")
            logger.info(f"Total runtime: {stats['runtime_formatted']}")
            logger.info(f"Words processed: {stats['processed_count']}")
            logger.info(
                f"Processing rate: {stats['processing_rate_per_minute']:.1f} words/minute"
            )
            logger.info(f"Queue size: {stats['queue_size']}")
            logger.info(f"Unique words: {stats['total_unique_words']}")
            logger.info(f"Errors: {stats['error_count']}")

            # Print table with performance metrics
            headers = ["Metric", "Value"]
            rows = [
                [
                    "Processing time (avg)",
                    f"{stats['avg_processing_time']*1000:.1f} ms",
                ],
                ["Words processed", f"{stats['processed_count']}"],
                ["Queue depth", f"{stats['queue_size']}"],
                ["Productivity", f"{stats['productivity_metric']:.2f}"],
                ["Errors", f"{stats['error_count']}"],
            ]
            print_table(headers, rows, title="Performance Summary")


if __name__ == "__main__":
    main()
