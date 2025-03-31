import datetime
import logging
import os
import signal
import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, TypedDict, Union

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
    details: Dict[str, any]


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
        sleep_interval: float = 1.0,
        error_backoff_factor: float = 1.5,
        max_errors_per_minute: int = 10,
        daemon: bool = True,
        logger: Optional[logging.Logger] = None,
        result_callback: Optional[Callable[[ProcessingResult], None]] = None,
    ):
        """
        Initialize a worker thread for background word processing.

        Args:
            parser_refiner: Parser component that processes words
            queue_manager: Queue providing words to process
            sleep_interval: Seconds to sleep when queue is empty
            error_backoff_factor: Multiplier for sleep time after errors
            max_errors_per_minute: Errors per minute before forced pause
            daemon: Whether thread runs as daemon (terminates with main)
            logger: Optional logger for status messages
            result_callback: Optional callback for processing results
        """
        super().__init__(daemon=daemon)
        self.parser_refiner = parser_refiner
        self.queue_manager = queue_manager
        self.base_sleep_interval = sleep_interval
        self.sleep_interval = sleep_interval
        self.error_backoff_factor = error_backoff_factor
        self.max_errors_per_minute = max_errors_per_minute
        self.result_callback = result_callback

        # Setup logging
        self.logger = logger or logging.getLogger(__name__)

        # Internal state tracking
        self._state = WorkerState.INITIALIZING
        self._state_lock = threading.RLock()
        self._stats = WorkerStatistics()
        self._recent_errors: List[float] = []  # Timestamps of recent errors
        self._pause_until: float = 0
        self._stop_requested = threading.Event()

        # Performance tracking
        self._processed_terms: Set[str] = set()
        self._initial_queue_size = 0
        self._productivity_metric = 0.0

        # Register signal handlers if this is the main thread
        if threading.current_thread() is threading.main_thread():
            self._register_signal_handlers()

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
            self._stats.add_event(
                {
                    "timestamp": time.time(),
                    "event_type": "state_change",
                    "details": {
                        "old_state": old_state.name,
                        "new_state": new_state.name,
                    },
                }
            )

    def get_statistics(self) -> Dict[str, Union[int, float, str, None, List, Dict]]:
        """
        Get worker statistics as a dictionary for monitoring.

        Returns:
            Dictionary with runtime statistics and state information
        """
        runtime = self._stats.runtime_seconds
        queue_size = self.queue_manager.size()
        unique_words = len(list(self.queue_manager.iter_seen()))

        # Calculate productivity metric: processed words relative to queue growth
        if self._initial_queue_size > 0:
            words_processed = self._stats.processing_count
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
            "processed_count": self._stats.processing_count,
            "error_count": self._stats.error_count,
            "empty_queue_count": self._stats.empty_queue_count,
            "processing_rate_per_minute": self._stats.processing_rate,
            "avg_processing_time": self._stats.average_processing_time,
            "queue_size": queue_size,
            "total_unique_words": unique_words,
            "last_processed": self._stats.last_processed,
            "last_error": self._stats.last_error,
            "idle_seconds": self._stats.idle_seconds,
            "productivity_metric": self._productivity_metric,
            "recent_events": self._stats.recent_events[-5:],  # Last 5 events
            "performance": {
                "mean_processing_time": self._stats.average_processing_time,
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
        """
        Main worker loop that processes queued words until stopped.

        The loop handles state transitions, error recovery, and
        implements dynamic sleep intervals based on queue status.
        """
        self._stats.start_time = time.time()
        self._stats.last_active = time.time()
        self._initial_queue_size = self.queue_manager.size()
        self.state = WorkerState.RUNNING

        while not self._stop_requested.is_set():
            try:
                # Handle paused state
                if self.state == WorkerState.PAUSED:
                    if time.time() < self._pause_until:
                        time.sleep(min(1.0, self._pause_until - time.time()))
                        continue
                    else:
                        # Auto-resume after pause expires
                        self.state = WorkerState.RUNNING
                        self.sleep_interval = self.base_sleep_interval

                # Process next word if available
                self._process_next_word()

            except Exception as e:
                self._handle_unexpected_error(e)

        self.state = WorkerState.STOPPED
        self.logger.info("Worker stopped gracefully")

        # Record final event
        self._stats.add_event(
            {
                "timestamp": time.time(),
                "event_type": "worker_stopped",
                "details": {
                    "total_processed": self._stats.processing_count,
                    "runtime_seconds": self._stats.runtime_seconds,
                },
            }
        )

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
            self._stats.last_active = start_time
            self.logger.debug(f"Processing: '{term}'")

            # Process the word and track result
            initial_queue_size = self.queue_manager.size()
            success = self.parser_refiner.process_word(term)
            end_time = time.time()
            processing_duration = end_time - start_time

            # Get relationships count from database
            relationships_count = 0
            new_terms_count = self.queue_manager.size() - initial_queue_size

            word_entry = self.parser_refiner.db_manager.get_word_if_exists(term)
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
                self._stats.processing_count += 1
                self._stats.last_processed = term
                self._stats.record_processing_time(processing_duration)
                # Reset sleep interval after successful processing
                self.sleep_interval = self.base_sleep_interval

                # Record successful processing event
                self._stats.add_event(
                    {
                        "timestamp": time.time(),
                        "event_type": "word_processed",
                        "term": term,
                        "duration": processing_duration,
                        "details": {
                            "relationships": relationships_count,
                            "new_terms": new_terms_count,
                        },
                    }
                )
            else:
                # Non-critical processing failure
                result["error"] = f"Failed to process '{term}'"
                self.logger.warning(f"Failed to process '{term}'")

                # Record failure event
                self._stats.add_event(
                    {
                        "timestamp": time.time(),
                        "event_type": "processing_failure",
                        "term": term,
                        "details": {"reason": "Failed to process word"},
                    }
                )

            # Call callback if provided
            if self.result_callback:
                try:
                    self.result_callback(result)
                except Exception as e:
                    self.logger.error(f"Error in result callback: {str(e)}")

        except EmptyQueueError:
            # Queue is empty, apply backoff sleep
            self._stats.empty_queue_count += 1
            self._stats.add_event(
                {"timestamp": time.time(), "event_type": "empty_queue", "details": {}}
            )
            self._sleep_with_interruption(self.sleep_interval)

    def _handle_unexpected_error(self, error: Exception) -> None:
        """
        Handle unexpected errors during processing with escalating responses.

        Args:
            error: The exception that was caught
        """
        self._stats.error_count += 1
        self._stats.last_error = str(error)
        error_trace = traceback.format_exc()
        self.logger.error(f"Unexpected error: {error}\n{error_trace}")

        # Record error event
        self._stats.add_event(
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
            # Apply exponential backoff for individual errors
            self.sleep_interval = min(
                30.0, self.sleep_interval * self.error_backoff_factor
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
        self._stats.add_event(
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
        Pause the worker for a specified duration or indefinitely.

        Args:
            seconds: Duration to pause in seconds, or None for indefinite
        """
        if seconds is not None:
            self._pause_until = time.time() + seconds
            self.logger.info(f"Pausing worker for {seconds} seconds")
        else:
            self._pause_until = float("inf")
            self.logger.info("Pausing worker indefinitely")

        self.state = WorkerState.PAUSED

    def resume(self) -> None:
        """Resume a paused worker."""
        if self.state == WorkerState.PAUSED:
            self.logger.info("Resuming worker")
            self._pause_until = 0
            self.state = WorkerState.RUNNING
            self.sleep_interval = self.base_sleep_interval


def create_demo_files(data_dir: Path) -> None:
    """
    Create sample dictionary files for demonstration.

    Args:
        data_dir: Directory to create files in
    """
    # Create sample dictionary files
    with open(data_dir / "openthesaurus.jsonl", "w") as f:
        f.write('{"words": ["algorithm"], "synonyms": ["procedure", "method"]}\n')
        f.write(
            '{"words": ["language"], "synonyms": ["tongue", "speech", "dialect"]}\n'
        )
        f.write(
            '{"words": ["computer"], "synonyms": ["machine", "processor", "device"]}\n'
        )
        f.write('{"words": ["network"], "synonyms": ["web", "grid", "system"]}\n')
        f.write(
            '{"words": ["data"], "synonyms": ["information", "facts", "figures"]}\n'
        )

    with open(data_dir / "odict.json", "w") as f:
        f.write(
            """{
            "algorithm": {"definition": "A step-by-step procedure", "examples": ["Sorting algorithms are fundamental."]},
            "language": {"definition": "A system of communication", "examples": ["English is a global language."]},
            "computer": {"definition": "An electronic device for processing data", "examples": ["The computer crashed."]},
            "data": {"definition": "Facts and statistics collected together", "examples": ["The data shows an increasing trend."]},
            "network": {"definition": "A group of interconnected systems", "examples": ["The computer network spans multiple buildings."]}
        }"""
        )

    with open(data_dir / "opendict.json", "w") as f:
        f.write("{}")

    with open(data_dir / "thesaurus.jsonl", "w") as f:
        f.write('{"word": "algorithm", "synonyms": ["process", "routine"]}\n')
        f.write('{"word": "language", "synonyms": ["communication", "expression"]}\n')
        f.write('{"word": "program", "synonyms": ["application", "software"]}\n')
        f.write('{"word": "function", "synonyms": ["procedure", "routine"]}\n')


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
        print(f"\n{title}")
        print("=" * len(title))

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
    import tempfile

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
        enable_model=False,  # Disable model for demo
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
            logger.debug(
                f"✓ Processed '{result['term']}' in {result['duration']:.3f}s - "
                f"{result['relationships_count']} relationships, {result['new_terms_count']} new terms"
            )

    logger.info(f"Seeding queue with {len(seed_words)} initial words...")
    for word in seed_words:
        queue_manager.enqueue(word)

    # Create and start worker
    logger.info("Creating worker thread...")
    worker = WordForgeWorker(
        parser_refiner,
        queue_manager,
        sleep_interval=0.05,  # Faster cycling for demo
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
            # Sleep for the update interval or until end time
            remaining = min(update_interval, end_time - time.time())
            if remaining <= 0:
                break
            time.sleep(remaining)

            # Show stats
            stats = worker.get_statistics()
            logger.info(worker.formatted_statistics())

            # Show progress bar
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / run_seconds)
            logger.info(f"Progress: {create_progress_bar(elapsed, run_seconds)}")

            # Check processing milestones
            processed_count = stats["processed_count"]
            while (
                next_milestone_idx < len(milestones)
                and processed_count >= milestones[next_milestone_idx]
            ):
                logger.info(
                    f"🏆 Milestone reached: {milestones[next_milestone_idx]} words processed!"
                )
                next_milestone_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Stopping gracefully...")

    finally:
        # Request worker to stop
        logger.info("Stopping worker...")
        worker.request_stop()
        worker.join(timeout=5.0)

        # Final statistics
        final_stats = worker.get_statistics()

        # Print summary table
        print("\n" + "=" * 60)
        print("📊 WORD FORGE WORKER STATISTICS 📊".center(60))
        print("=" * 60)

        # Print summary statistics as a table
        headers = ["Metric", "Value"]
        rows = [
            ["Runtime", final_stats["runtime_formatted"]],
            ["Words processed", final_stats["processed_count"]],
            [
                "Processing rate",
                f"{final_stats['processing_rate_per_minute']:.2f} words/min",
            ],
            [
                "Avg processing time",
                f"{final_stats['avg_processing_time']*1000:.2f} ms",
            ],
            ["Queue size", final_stats["queue_size"]],
            ["Unique words", final_stats["total_unique_words"]],
            ["Errors", final_stats["error_count"]],
            ["Empty queue events", final_stats["empty_queue_count"]],
            ["Worker state", final_stats["state"]],
        ]
        print_table(headers, rows)

        # Show a sample of processed words from the database
        processed_count = min(10, final_stats["processed_count"])
        if processed_count > 0:
            logger.info("\nSample of processed words:")
            seen_words = list(worker._processed_terms)
            sample_words = seen_words[:processed_count]

            # Create table rows for processed words
            word_headers = [
                "Word",
                "Relationships",
                "Part of Speech",
                "Definition Excerpt",
            ]
            word_rows = []

            for word in sample_words:
                word_entry = db_manager.get_word_if_exists(word)
                if word_entry:
                    rel_count = len(word_entry["relationships"])
                    pos = word_entry.get("part_of_speech", "")
                    definition = word_entry.get("definition", "")
                    if definition and len(definition) > 40:
                        definition = definition[:37] + "..."

                    word_rows.append([word, rel_count, pos, definition])

            if word_rows:
                print_table(word_headers, word_rows, "Processed Words Sample")

        logger.info(f"\nTemporary files at: {temp_dir}")
        logger.info("Demo complete!")


if __name__ == "__main__":
    main()
