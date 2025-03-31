import logging
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Literal, Optional, TypedDict

from word_forge.database.db_manager import DBManager
from word_forge.graph.graph_manager import GraphManager


class WorkerStatus(TypedDict):
    """Type definition for worker status information."""

    running: bool
    update_count: int
    error_count: int
    last_update: Optional[float]
    uptime: Optional[float]
    state: Literal["running", "stopped", "error"]


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
        poll_interval: float = 30.0,
        output_path: str = "data/word_graph.gexf",
        visualization_path: str = "data/graph_visualization.html",
        daemon: bool = True,
    ) -> None:
        """
        Initialize the graph worker with configuration parameters.

        Args:
            graph_manager: Graph manager instance to perform operations
            poll_interval: Seconds between update cycles
            output_path: Path where the graph will be saved
            visualization_path: Path where visualization will be saved
            daemon: Whether thread should be daemonic (auto-terminate when main exits)
        """
        super().__init__(daemon=daemon)
        self.graph_manager = graph_manager
        self.poll_interval = poll_interval
        self.output_path = output_path
        self.visualization_path = visualization_path
        self._stop_flag = False
        self._last_update: Optional[float] = None
        self._start_time: Optional[float] = None
        self._update_count = 0
        self._error_count = 0
        self._status_lock = threading.RLock()
        self.logger = logging.getLogger(__name__)

    def run(self) -> None:
        """
        Main execution loop that periodically updates and saves the graph.

        Updates occur at intervals defined by poll_interval. All exceptions
        are caught to prevent thread termination, logged with traceback
        information, and the worker continues to the next cycle.
        """
        self._start_time = time.time()
        self.logger.info(
            f"GraphWorker started: interval={self.poll_interval}s, output={self.output_path}"
        )

        while not self._stop_flag:
            try:
                self._ensure_output_directories()
                self._update_and_save_graph()
                self._generate_visualization()

                # Track successful updates
                with self._status_lock:
                    self._last_update = time.time()
                    self._update_count += 1

                self.logger.debug(
                    f"Graph updated and saved to {self.output_path} (update #{self._update_count})"
                )

            except Exception as e:
                with self._status_lock:
                    self._error_count += 1
                self.logger.error(f"Error updating graph: {str(e)}")
                self.logger.debug(f"Traceback: {traceback.format_exc()}")

            # Sleep until next cycle
            time.sleep(self.poll_interval)

        self.logger.info(
            f"GraphWorker stopped after {self._update_count} updates "
            f"with {self._error_count} errors"
        )

    def _ensure_output_directories(self) -> None:
        """Create output directories if they don't exist."""
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.visualization_path).parent.mkdir(parents=True, exist_ok=True)

    def _update_and_save_graph(self) -> None:
        """Update the graph and save it to disk."""
        self.graph_manager.build_graph()
        self.graph_manager.save_to_gexf(self.output_path)

    def _generate_visualization(self) -> None:
        """Generate graph visualization if the graph has nodes."""
        if self.graph_manager.get_node_count() > 0:
            try:
                self.graph_manager.visualize(output_path=self.visualization_path)
                self.logger.debug(
                    f"Graph visualization saved to {self.visualization_path}"
                )
            except Exception as viz_error:
                self.logger.warning(
                    f"Failed to generate visualization: {str(viz_error)}"
                )

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
            - state: Current worker state ("running", "stopped", or "error")
        """
        with self._status_lock:
            uptime = None
            if self._start_time:
                uptime = time.time() - self._start_time

            state: Literal["running", "stopped", "error"] = "stopped"
            if self.is_alive() and not self._stop_flag:
                state = "error" if self._error_count > self._update_count else "running"

            status: WorkerStatus = {
                "running": self.is_alive() and not self._stop_flag,
                "update_count": self._update_count,
                "error_count": self._error_count,
                "last_update": self._last_update,
                "uptime": uptime,
                "state": state,
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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    logger = logging.getLogger("graph_worker_demo")
    logger.info("Starting graph worker demonstration")

    # Initialize database and graph managers
    db_path = "word_forge.sqlite"
    logger.info(f"Using database: {db_path}")

    try:
        db_manager = DBManager(db_path=db_path)
        graph_manager = GraphManager(db_manager)

        # Ensure DB has tables and sample data
        db_manager._create_tables()  # Note: using protected method for compatibility
        if graph_manager.ensure_sample_data():
            logger.info("Added sample data to database")

        # Configure and start the worker
        worker = GraphWorker(
            graph_manager=graph_manager,
            poll_interval=10.0,  # More frequent updates for demo
            output_path="data/word_graph.gexf",
            visualization_path="data/graph_visualization.html",
        )

        logger.info("Starting graph worker")
        worker.start()

        # Check status periodically
        for _ in range(3):
            time.sleep(5)
            status = worker.get_status()
            logger.info(f"Worker status: {status}")

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


if __name__ == "__main__":
    main()
