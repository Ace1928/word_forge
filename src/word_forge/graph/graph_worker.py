import logging
import os
import random
import sys
import threading
import time
import traceback
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Protocol, TypedDict, final

from word_forge.config import config
from word_forge.database.db_manager import DBManager
from word_forge.graph.graph_manager import GraphManager


class WorkerState(Enum):
    """Defined states for the graph worker."""

    RUNNING = auto()
    STOPPED = auto()
    ERROR = auto()
    PAUSED = auto()  # Added for more granular state control

    def __str__(self) -> str:
        """Return the lowercase name of the state."""
        return self.name.lower()


class WorkerStatus(TypedDict):
    """Type definition for worker status information."""

    running: bool
    update_count: int
    error_count: int
    last_update: Optional[float]
    uptime: Optional[float]
    state: str  # String representation of WorkerState
    last_error: Optional[str]  # Added to track last error message


class GraphError(Exception):
    """Base exception for graph operation errors."""

    pass


class GraphSaveError(GraphError):
    """Raised when a graph save operation fails."""

    pass


class GraphUpdateError(GraphError):
    """Raised when a graph update operation fails."""

    pass


class GraphVisualizationError(GraphError):
    """Raised when a graph visualization operation fails."""

    pass


class GraphDirectoryError(GraphError):
    """Raised when a directory operation fails."""

    pass


class GraphWorkerInterface(Protocol):
    """Protocol defining the required interface for a graph worker."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_status(self) -> WorkerStatus: ...
    def is_alive(self) -> bool: ...
    def pause(self) -> None: ...  # Added for completeness
    def resume(self) -> None: ...  # Added for completeness


@final
class GraphWorker(threading.Thread):
    """
    Background worker that maintains the lexical graph representation.

    Periodically rebuilds or updates the in-memory graph and saves it to disk
    for visualization and further queries. Thread-safe operations ensure graph
    integrity across concurrent access patterns.

    Attributes:
        graph_manager: Manager handling graph operations
        poll_interval: Seconds between graph update cycles
        output_path: Path where graph will be saved
        visualization_path: Path where visualization will be saved
    """

    def __init__(
        self,
        graph_manager: GraphManager,
        poll_interval: Optional[float] = None,
        output_path: Optional[str] = None,
        visualization_path: Optional[str] = None,
        daemon: bool = True,
    ) -> None:
        """
        Initialize the graph worker with configuration parameters.

        Args:
            graph_manager: Graph manager instance to perform operations
            poll_interval: Seconds between update cycles (defaults to config)
            output_path: Path where the graph will be saved (defaults to config)
            visualization_path: Path where visualization will be saved (defaults to config)
            daemon: Whether thread should be daemonic (auto-terminate when main exits)
        """
        super().__init__(daemon=daemon)
        self.graph_manager = graph_manager

        # Use config values if parameters not provided
        self.poll_interval = (
            poll_interval or config.graph.animation_duration_ms / 1000 or 30.0
        )

        # Ensure proper paths for output and visualization
        self.output_path = output_path or str(
            config.graph.get_export_filepath("lexical_graph")
        )

        # For visualization, ensure we have a directory and filename
        vis_path = Path(visualization_path or config.graph.visualization_path)
        if vis_path.suffix == "":  # If it's a directory, add a default filename
            vis_path = vis_path / "lexical_graph.html"
        self.visualization_path = str(vis_path)

        self._stop_flag = False
        self._pause_flag = False
        self._last_update: Optional[float] = None
        self._start_time: Optional[float] = None
        self._update_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._current_state = WorkerState.STOPPED
        self._status_lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

    def pause(self) -> None:
        """Pause worker execution without stopping the thread."""
        with self._status_lock:
            self._pause_flag = True
            self._current_state = WorkerState.PAUSED
        self.logger.info("GraphWorker paused")

    def resume(self) -> None:
        """Resume worker execution after being paused."""
        with self._status_lock:
            self._pause_flag = False
            self._current_state = WorkerState.RUNNING
        self.logger.info("GraphWorker resumed")

    def run(self) -> None:
        """
        Main execution loop that periodically updates and saves the graph.

        Updates occur at intervals defined by poll_interval. All exceptions
        are caught to prevent thread termination, logged with traceback
        information, and the worker continues to the next cycle.
        """
        with self._status_lock:
            self._start_time = time.time()
            self._current_state = WorkerState.RUNNING

        self.logger.info(
            f"GraphWorker started: interval={self.poll_interval}s, output={self.output_path}"
        )

        while not self._stop_flag:
            if self._pause_flag:
                time.sleep(1.0)  # Reduced CPU usage while paused
                continue

            try:
                self._execute_update_cycle()
                time.sleep(self.poll_interval)
            except Exception as e:
                self._handle_execution_error(e)
                time.sleep(max(1.0, self.poll_interval / 2))  # Reduced sleep on error

        with self._status_lock:
            self._current_state = WorkerState.STOPPED

        self.logger.info(
            f"GraphWorker stopped after {self._update_count} updates "
            f"with {self._error_count} errors"
        )

    def _execute_update_cycle(self) -> None:
        """Execute a complete update cycle: prepare, update, save, visualize."""
        try:
            self._ensure_output_directories()
        except Exception as e:
            raise GraphDirectoryError(f"Failed to create directories: {str(e)}") from e

        try:
            # Check if database tables exist and are accessible before updating
            if not self._verify_database_tables():
                self.logger.warning(
                    "Required database tables not found. Graph updates paused until data becomes available."
                )
                # Set a longer backoff period before trying again
                self._trigger_backoff(30.0)  # Wait 30 seconds before retry
                return

            self._update_graph()
        except Exception as e:
            # Record error and trigger backoff
            self._handle_graph_error(e)
            return

        try:
            self._save_graph()
        except Exception as e:
            raise GraphSaveError(f"Failed to save graph: {str(e)}") from e

        try:
            self._generate_visualization()
        except Exception as e:
            # Visualization errors are non-fatal
            self.logger.warning(f"Failed to generate visualization: {str(e)}")

        # Track successful updates
        with self._status_lock:
            self._last_update = time.time()
            self._update_count += 1
            self._current_state = WorkerState.RUNNING
            self._last_error = None  # Clear error state after successful update
            self._error_backoff = self.update_interval  # Reset backoff

        self.logger.debug(
            f"Graph updated and saved to {self.output_path} (update #{self._update_count})"
        )

    def _verify_database_tables(self) -> bool:
        """Verify that required database tables exist.

        Returns:
            bool: True if tables exist, False otherwise
        """
        try:
            # Use graph_manager's connection to verify tables
            return self.graph_manager.verify_database_tables()
        except Exception as e:
            self.logger.error(f"Database verification failed: {e}")
            return False

    def _handle_graph_error(self, error: Exception) -> None:
        """Handle graph update errors with exponential backoff.

        Args:
            error: The exception that occurred
        """
        self._error_count += 1

        # Store the error
        with self._status_lock:
            self._last_error = str(error)
            self._current_state = WorkerState.ERROR

        # Implement exponential backoff
        self._trigger_backoff()

        # Log error (but prevent error spam)
        if self._error_count <= 3 or self._error_count % 10 == 0:
            self.logger.error(f"GraphUpdateError updating graph: {error}")
        else:
            self.logger.debug(f"GraphUpdateError updating graph: {error}")

    def _trigger_backoff(self, initial_delay: Optional[float] = None) -> None:
        """Implement exponential backoff for error recovery.

        Args:
            initial_delay: Optional custom initial delay
        """
        if initial_delay is not None:
            self._error_backoff = initial_delay
        else:
            # Exponential backoff with jitter - increase delay with each consecutive error
            self._error_backoff = min(
                60.0,  # Cap at 60 seconds max
                self._error_backoff * 1.5 + random.uniform(0, 1),
            )

        self.logger.info(
            f"Graph updates paused for {self._error_backoff:.1f}s (backoff)"
        )

    def _handle_execution_error(self, error: Exception) -> None:
        """Process execution errors with appropriate logging and state updates."""
        error_message = str(error)
        error_type = type(error).__name__

        with self._status_lock:
            self._error_count += 1
            self._current_state = WorkerState.ERROR
            self._last_error = f"{error_type}: {error_message}"

        self.logger.error(f"{error_type} updating graph: {error_message}")
        self.logger.debug(f"Traceback: {traceback.format_exc()}")

    def _ensure_output_directories(self) -> None:
        """Create output directories if they don't exist."""
        # Fixed: Use os.path to check if directory exists before creating
        # to avoid errors with existing directories
        for path in [self.output_path, self.visualization_path]:
            dir_path = os.path.dirname(path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

    def _update_graph(self) -> None:
        """Update the graph data structure."""
        self.graph_manager.build_graph()

    def _save_graph(self) -> None:
        """Save the graph to a file."""
        self.graph_manager.save_to_gexf(self.output_path)

    def _generate_visualization(self) -> None:
        """Generate graph visualization if the graph has nodes."""
        if self.graph_manager.get_node_count() > 0:
            self.graph_manager.visualize(output_path=self.visualization_path)
            self.logger.debug(f"Graph visualization saved to {self.visualization_path}")

    def stop(self) -> None:
        """
        Signal the worker to stop after completing current operations.

        Sets the internal stop flag that will be detected at the next loop
        iteration. The worker thread will terminate after completing any
        in-progress graph operations.
        """
        self.logger.info("GraphWorker stopping...")
        self._stop_flag = True

    def get_status(self) -> WorkerStatus:
        """
        Return the current status of the graph worker.

        Returns:
            Dictionary containing operational metrics including:
            - running: Whether the worker is active
            - update_count: Number of successful updates
            - error_count: Number of encountered errors
            - last_update: Timestamp of last successful update
            - uptime: Seconds since thread start if running
            - state: Current worker state ("running", "stopped", "paused", or "error")
            - last_error: Most recent error message if any
        """
        with self._status_lock:
            uptime = None
            if self._start_time:
                uptime = time.time() - self._start_time

            status: WorkerStatus = {
                "running": self.is_alive()
                and not self._stop_flag
                and not self._pause_flag,
                "update_count": self._update_count,
                "error_count": self._error_count,
                "last_update": self._last_update,
                "uptime": uptime,
                "state": str(self._current_state),
                "last_error": self._last_error,
            }

            return status


def main() -> None:
    """
    Demonstrate usage of the GraphWorker to maintain a lexical graph.

    Creates a worker thread that periodically updates the graph and saves
    both a GEXF file for storage and an HTML visualization for viewing.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=config.logging.format,
        stream=sys.stdout,
    )

    logger = logging.getLogger("graph_worker_demo")
    logger.info("Starting graph worker demonstration")

    # Initialize database and graph managers
    db_path = config.database.db_path
    logger.info(f"Using database: {db_path}")

    worker = None
    try:
        db_manager = DBManager(db_path=db_path)
        graph_manager = GraphManager(db_manager)

        # Ensure DB has tables and sample data
        db_manager.create_tables()  # Note: using protected method for compatibility
        if graph_manager.ensure_sample_data():
            logger.info("Added sample data to database")

        # Set up paths for the worker
        export_path = Path(config.graph.default_export_path) / "lexical_graph.gexf"
        vis_path = Path(config.graph.visualization_path) / "lexical_graph.html"

        # Pre-create directories to avoid race conditions
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        os.makedirs(os.path.dirname(vis_path), exist_ok=True)

        # Configure and start the worker
        worker = GraphWorker(
            graph_manager=graph_manager,
            poll_interval=10.0,  # More frequent updates for demo
            output_path=str(export_path),
            visualization_path=str(vis_path),
        )

        logger.info("Starting graph worker")
        worker.start()

        # Check status periodically
        for _ in range(3):
            time.sleep(5)
            status = worker.get_status()
            logger.info(f"Worker status: {status}")

            # If in error state, show detailed error
            if status["state"] == "error" and status["last_error"]:
                logger.info(f"Error details: {status['last_error']}")

        # Stop the worker
        logger.info("Stopping worker")
        worker.stop()
        worker.join(timeout=15.0)

        final_status = worker.get_status()
        logger.info(f"Final worker status: {final_status}")
        logger.info(
            f"Graph saved to {worker.output_path}, "
            f"visualization available at {worker.visualization_path}"
        )

    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        logger.debug(traceback.format_exc())
    finally:
        # Ensure worker is properly stopped if an exception occurs
        if worker and worker.is_alive():
            logger.info("Stopping worker due to exception")
            worker.stop()
            worker.join(timeout=5.0)


if __name__ == "__main__":
    main()
