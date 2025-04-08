"""
Worker Manager Module for Word Forge.

Orchestrates the lifecycle and coordination of various background worker threads,
ensuring robust, concurrent, and efficient processing of lexical, conversational,
and analytical tasks. This manager provides a unified interface for controlling
and monitoring all system workers.

Architecture:
    ┌────────────────────┐
    │    WorkerManager   │
    └──────────┬─────────┘
               │ Manages
    ┌──────────┴──────────┐
    │      Workers        │
    └─────────────────────┘
    ┌────────┬────────┬────────┬────────┬────────┬────────┐
    │Conv.   │ DB     │ Vector │ Emotion│ Parser │ Graph  │
    │Worker  │ Worker │ Worker │ Worker │ Worker │ Worker │
    └────────┴────────┴────────┴────────┴────────┴────────┘

Key Responsibilities:
- Initialize and register worker instances.
- Provide unified lifecycle control (start, stop, pause, resume, restart).
- Aggregate status and metrics from all managed workers.
- Handle worker errors and potentially implement recovery strategies.
- Ensure thread-safe operations for managing worker states.
"""

from __future__ import annotations

import _thread
import logging
import threading
import time
import types
from typing import Any, Dict, Optional, Protocol, Union, final

# Import worker types - use Protocols for loose coupling
from word_forge.conversation.conversation_worker import (
    ConversationWorker,
    ConversationWorkerInterface,
    ConversationWorkerStatus,
)
from word_forge.database.database_worker import (
    DatabaseWorker,
    DatabaseWorkerInterface,
    DBWorkerStatus,
)
from word_forge.emotion.emotion_worker import (
    EmotionWorker,
    EmotionWorkerInterface,
    EmotionWorkerStatus,
)
from word_forge.graph.graph_worker import GraphWorker, GraphWorkerInterface
from word_forge.graph.graph_worker import WorkerStatus as GraphWorkerStatus


# Assuming ParserWorker and VectorWorker exist and have similar interfaces
# Define placeholder protocols if actual implementations are not available yet
class ParserWorkerInterface(Protocol):
    """Protocol for ParserWorker."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def restart(self) -> None: ...
    def get_status(self) -> Dict[str, Any]: ...  # Placeholder status type
    def get_metrics(self) -> Dict[str, Any]: ...  # Added get_metrics
    def is_alive(self) -> bool: ...


class VectorWorkerInterface(Protocol):
    """Protocol for VectorWorker."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def restart(self) -> None: ...
    def get_status(self) -> Dict[str, Any]: ...  # Placeholder status type
    def get_metrics(self) -> Dict[str, Any]: ...  # Added get_metrics
    def is_alive(self) -> bool: ...
    def run(self) -> None: ...


# Define a union type for all possible worker interfaces
WorkerInterface = Union[
    ConversationWorkerInterface,
    DatabaseWorkerInterface,
    VectorWorkerInterface,
    EmotionWorkerInterface,
    ParserWorkerInterface,
    GraphWorkerInterface,
]

# Define a union type for all possible worker status dictionaries
WorkerStatusType = Union[
    ConversationWorkerStatus,
    DBWorkerStatus,
    EmotionWorkerStatus,
    GraphWorkerStatus,
    Dict[str, Any],  # Placeholder for Parser/Vector workers
]

# Configure module logger
logger = logging.getLogger(__name__)


class WorkerManagerError(Exception):
    """Base exception for worker manager errors."""

    pass


class WorkerRegistrationError(WorkerManagerError):
    """Raised when registering a worker fails."""

    pass


class WorkerControlError(WorkerManagerError):
    """Raised when controlling a worker (start, stop, etc.) fails."""

    pass


@final
class WorkerManager:
    """
    Central orchestrator for all background worker threads in Word Forge.

    Manages the lifecycle, status reporting, and coordination of workers
    responsible for conversation processing, database maintenance, vector
    embedding, emotion analysis, parsing, and graph generation.

    Ensures robust operation through thread-safe management and error handling.
    Adheres to Eidosian principles of precision, integration, and autonomy.
    """

    def __init__(self, enable_logging: bool = True) -> None:
        """
        Initialize the WorkerManager.

        Args:
            enable_logging: Whether to enable logging for the manager.
        """
        self._workers: Dict[str, WorkerInterface] = {}
        self._worker_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        self.logger = logger if enable_logging else None
        self._running = False

        if self.logger:
            self.logger.info("WorkerManager initialized")

    def register_worker(self, name: str, worker_instance: WorkerInterface) -> None:
        """
        Register a worker instance with the manager.

        Args:
            name: A unique name for the worker (e.g., "conversation", "database").
            worker_instance: The worker instance to manage.

        Raises:
            WorkerRegistrationError: If a worker with the same name is already registered
                                     or if the instance is not a valid worker thread.
        """
        with self._lock:
            if name in self._workers:
                raise WorkerRegistrationError(
                    f"Worker with name '{name}' already registered."
                )

            # Basic validation: Check if it's a Thread instance and has required methods
            if not isinstance(worker_instance, threading.Thread):
                raise WorkerRegistrationError(
                    f"Worker '{name}' must be an instance of threading.Thread."
                )
            required_methods = ["start", "stop", "is_alive", "get_status"]
            if not all(hasattr(worker_instance, method) for method in required_methods):
                raise WorkerRegistrationError(
                    f"Worker '{name}' is missing required methods."
                )

            self._workers[name] = worker_instance
            self._worker_threads[name] = worker_instance  # Store the thread instance

            if self.logger:
                self.logger.info(
                    f"Registered worker: '{name}' (Type: {type(worker_instance).__name__})"
                )

            # If the manager is already running, start the newly registered worker
            if self._running:
                try:
                    self.start_worker(name)
                except WorkerControlError as e:
                    if self.logger:
                        self.logger.error(
                            f"Failed to auto-start newly registered worker '{name}': {e}"
                        )

    def unregister_worker(self, name: str) -> None:
        """
        Unregister a worker instance. Stops the worker if it's running.

        Args:
            name: The name of the worker to unregister.

        Raises:
            KeyError: If the worker name is not found.
        """
        with self._lock:
            if name not in self._workers:
                raise KeyError(f"Worker '{name}' not found.")

            worker = self._workers[name]
            if worker.is_alive():
                try:
                    self.stop_worker(name, wait=True, timeout=5.0)
                except WorkerControlError as e:
                    if self.logger:
                        self.logger.warning(
                            f"Error stopping worker '{name}' during unregistration: {e}"
                        )

            del self._workers[name]
            del self._worker_threads[name]

            if self.logger:
                self.logger.info(f"Unregistered worker: '{name}'")

    def start_all(self) -> None:
        """Start all registered worker threads."""
        with self._lock:
            if self._running:
                if self.logger:
                    self.logger.warning("WorkerManager is already running.")
                return

            self._running = True
            if self.logger:
                self.logger.info("Starting all registered workers...")

            for name in list(self._workers.keys()):  # Iterate over a copy of keys
                try:
                    self.start_worker(name)
                except WorkerControlError as e:
                    if self.logger:
                        self.logger.error(f"Failed to start worker '{name}': {e}")
                except Exception as e:
                    if self.logger:
                        self.logger.error(
                            f"Unexpected error starting worker '{name}': {e}",
                            exc_info=True,
                        )

    def stop_all(self, wait: bool = True, timeout: float = 10.0) -> None:
        """
        Stop all registered worker threads gracefully.

        Args:
            wait: If True, wait for all threads to terminate.
            timeout: Maximum time in seconds to wait for each thread.
        """
        with self._lock:
            if not self._running:
                if self.logger:
                    self.logger.warning("WorkerManager is not running.")
                return

            self._running = False
            active_threads = []

            if self.logger:
                self.logger.info("Stopping all registered workers...")

            # Define explicit type annotation for active_threads
            active_threads: list[threading.Thread] = []

            # Stop each worker
            for name, worker in self._workers.items():
                try:
                    worker.stop()
                    if self.logger:
                        self.logger.info(f"Stopped worker '{name}'")

                    thread = self._worker_threads.get(name)
                    if thread and thread.is_alive():
                        active_threads.append(thread)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error stopping worker '{name}': {e}")

            # Wait for threads to terminate
            if wait and active_threads:
                if self.logger:
                    self.logger.info(
                        f"Waiting for {len(active_threads)} worker threads to terminate..."
                    )

                # Join threads with individual timeouts
                for thread in active_threads:
                    thread.join(timeout=timeout)

                # Check and warn for non-terminated threads
                non_terminated = [t for t in active_threads if t.is_alive()]
                for thread in non_terminated:
                    if self.logger:
                        self.logger.warning(
                            f"Worker thread {thread.name} did not terminate within timeout."
                        )

                if self.logger and not non_terminated:
                    self.logger.info("All workers stopped.")

    def pause_all(self) -> None:
        """Pause all workers that support pausing."""
        with self._lock:
            if not self._running:
                if self.logger:
                    self.logger.warning("Cannot pause workers, manager is not running.")
                return

            if self.logger:
                self.logger.info("Pausing all stoppable workers...")
            for name, worker in self._workers.items():
                if hasattr(worker, "pause"):
                    try:
                        worker.pause()
                        if self.logger:
                            self.logger.debug(f"Paused worker '{name}'")
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to pause worker '{name}': {e}")

    def resume_all(self) -> None:
        """Resume all paused workers."""
        with self._lock:
            if not self._running:
                if self.logger:
                    self.logger.warning(
                        "Cannot resume workers, manager is not running."
                    )
                return

            if self.logger:
                self.logger.info("Resuming all paused workers...")
            for name, worker in self._workers.items():
                if hasattr(worker, "resume"):
                    try:
                        worker.resume()
                        if self.logger:
                            self.logger.debug(f"Resumed worker '{name}'")
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to resume worker '{name}': {e}")

    def restart_all(self) -> None:
        """Restart all workers that support restarting."""
        with self._lock:
            if not self._running:
                if self.logger:
                    self.logger.warning(
                        "Cannot restart workers, manager is not running."
                    )
                return

            if self.logger:
                self.logger.info("Restarting all restartable workers...")
            for name, worker in self._workers.items():
                if hasattr(worker, "restart"):
                    try:
                        worker.restart()
                        if self.logger:
                            self.logger.debug(f"Restarted worker '{name}'")
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"Failed to restart worker '{name}': {e}")

    def start_worker(self, name: str) -> None:
        """Start a specific worker by name."""
        with self._lock:
            if name not in self._workers:
                raise WorkerControlError(f"Worker '{name}' not found.")

            worker = self._workers[name]
            thread = self._worker_threads[name]

            if thread.is_alive():
                if self.logger:
                    self.logger.warning(f"Worker '{name}' is already running.")
                return

            try:
                # Ensure worker state is ready before starting
                # Check specifically for threading.Event before calling clear
                if hasattr(worker, "_stop_event") and isinstance(
                    getattr(worker, "_stop_event", None), threading.Event
                ):
                    getattr(worker, "_stop_event").clear()
                if hasattr(worker, "_pause_event") and isinstance(
                    getattr(worker, "_pause_event", None), threading.Event
                ):
                    getattr(worker, "_pause_event").clear()

                # Re-create thread instance if it has terminated
                if not thread.is_alive() and getattr(thread, "_started", False):
                    # This assumes worker classes can be re-instantiated with same args
                    # A more robust solution might need worker factory functions
                    # For now, we'll log a warning and attempt restart if possible
                    if hasattr(worker, "restart"):
                        worker.restart()
                        if self.logger:
                            self.logger.info(
                                f"Restarted worker '{name}' via its restart method."
                            )
                        return
                    else:
                        if self.logger:
                            self.logger.warning(
                                f"Cannot restart terminated thread '{name}' directly. Re-registration might be needed."
                            )
                        # Attempt to start the existing instance if it supports it
                        try:
                            worker.start()
                            if self.logger:
                                self.logger.info(f"Started worker '{name}'")
                        except RuntimeError:  # Already started
                            if self.logger:
                                self.logger.warning(
                                    f"Worker '{name}' reported as not alive but start() failed."
                                )
                        except Exception as start_err:
                            raise WorkerControlError(
                                f"Failed to start worker '{name}': {start_err}"
                            ) from start_err
                        return

                # Start the thread
                worker.start()
                if self.logger:
                    self.logger.info(f"Started worker '{name}'")
            except RuntimeError as e:
                if "cannot start a thread twice" in str(e):
                    if self.logger:
                        self.logger.warning(
                            f"Attempted to start worker '{name}' twice."
                        )
                else:
                    raise WorkerControlError(
                        f"Failed to start worker '{name}': {e}"
                    ) from e
            except Exception as e:
                raise WorkerControlError(f"Failed to start worker '{name}': {e}") from e

    def stop_worker(
        self, name: str, wait: bool = False, timeout: float = 5.0
    ) -> Optional[threading.Thread]:
        """Stop a specific worker by name."""
        with self._lock:
            if name not in self._workers:
                raise WorkerControlError(f"Worker '{name}' not found.")

            worker = self._workers[name]
            thread = self._worker_threads[name]

            if not thread.is_alive():
                if self.logger:
                    self.logger.warning(f"Worker '{name}' is not running.")
                return None

            try:
                worker.stop()
                if self.logger:
                    self.logger.info(f"Stopped worker '{name}'")

                if wait:
                    thread.join(timeout=timeout)
                    if thread.is_alive():
                        if self.logger:
                            self.logger.warning(
                                f"Worker '{name}' did not terminate within {timeout}s."
                            )
                return thread
            except Exception as e:
                raise WorkerControlError(f"Failed to stop worker '{name}': {e}") from e

    def get_status(self) -> Dict[str, WorkerStatusType]:
        """
        Get the status of all registered workers.

        Returns:
            A dictionary mapping worker names to their status dictionaries.
        """
        statuses: Dict[str, WorkerStatusType] = {}
        with self._lock:
            for name, worker in self._workers.items():
                try:
                    # Handle potential worker errors or deadlocks
                    status = worker.get_status()
                    statuses[name] = status
                except Exception as e:
                    if self.logger:
                        self.logger.error(
                            f"Error getting status for worker '{name}': {e}"
                        )
                    statuses[name] = {
                        "running": False,
                        "state": "error",
                        "error": f"Status retrieval failed: {str(e)}",
                    }
        return statuses

    def get_worker_status(self, name: str) -> WorkerStatusType:
        """
        Get the status of a specific worker.

        Args:
            name: The name of the worker.

        Returns:
            The status dictionary for the specified worker.

        Raises:
            KeyError: If the worker name is not found.
            WorkerControlError: If status retrieval fails.
        """
        with self._lock:
            if name not in self._workers:
                raise KeyError(f"Worker '{name}' not found.")
            worker = self._workers[name]
            try:
                # Ensure get_status exists and is callable
                if hasattr(worker, "get_status") and callable(worker.get_status):
                    status = worker.get_status()
                    if "running" in status and "state" in status:
                        return status
                    else:
                        raise WorkerControlError(
                            f"Worker '{name}' returned invalid status format."
                        )
                else:
                    raise WorkerControlError(
                        f"Worker '{name}' does not have a get_status method."
                    )
            except Exception as e:
                raise WorkerControlError(
                    f"Failed to get status for worker '{name}': {e}"
                ) from e

    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics from all workers that support it.

        Returns:
            A dictionary mapping worker names to their metrics dictionaries.
        """
        metrics: Dict[str, Dict[str, Any]] = {}
        with self._lock:
            for name, worker in self._workers.items():
                if hasattr(worker, "get_metrics") and callable(worker.get_metrics):
                    try:
                        worker_metrics = worker.get_metrics()
                        metrics[name] = worker_metrics
                    except Exception as e:
                        if self.logger:
                            self.logger.error(
                                f"Failed to get metrics for worker '{name}': {e}"
                            )
                        metrics[name] = {"error": f"Metrics retrieval failed: {str(e)}"}
                else:
                    # Indicate that metrics are not available for this worker
                    metrics[name] = {"status": "Metrics not supported"}
        return metrics

    def is_running(self) -> bool:
        """Check if the WorkerManager is currently managing active workers."""
        with self._lock:
            return self._running

    def __enter__(self) -> "WorkerManager":
        """Enter context manager, starting all workers."""
        self.start_all()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[types.TracebackType],
    ) -> None:
        """Exit context manager, stopping all workers."""
        self.stop_all(wait=True)
        self.stop_all(wait=True)


# Example Usage (can be placed in a separate demo script or under if __name__ == "__main__":)
def worker_manager_demo():
    """Demonstrates the WorkerManager functionality."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # --- Assume necessary components are initialized ---
    # These would be properly initialized in a real application
    try:
        from word_forge.conversation.conversation_manager import ConversationManager
        from word_forge.database.database_manager import DBManager
        from word_forge.emotion.emotion_manager import EmotionManager
        from word_forge.graph.graph_manager import GraphManager
        from word_forge.parser.parser_refiner import ParserRefiner
        from word_forge.queue.queue_manager import QueueManager
        from word_forge.vectorizer.vector_store import VectorStore
        from word_forge.vectorizer.vector_worker import (
            TransformerEmbedder,  # Using TransformerEmbedder for demo
        )

        # Initialize Managers
        db_manager = DBManager("word_forge_demo.sqlite")
        db_manager.create_tables()  # Ensure tables exist
        emotion_manager = EmotionManager(db_manager)
        conversation_manager = ConversationManager(db_manager, emotion_manager)
        queue_manager_parser = QueueManager[str]()
        parser_refiner = ParserRefiner(db_manager, queue_manager_parser)
        # Use ConversationTask instead of ConversationWorkerStatus
        from word_forge.conversation.conversation_worker import ConversationTask

        queue_manager_conv = QueueManager[
            Union[ConversationTask, str, Dict[str, Any]]
        ]()
        # Removed unused vector_store variable
        # embedder = SimpleEmbedder()  # Uncomment when VectorWorker is implemented
        graph_manager = GraphManager(db_manager)

        # --- Initialize Workers ---
        conv_worker = ConversationWorker(
            parser_refiner,
            queue_manager_conv,
            conversation_manager,
            db_manager,
            emotion_manager,
        )
        db_worker = DatabaseWorker(db_manager)
        # vector_worker = VectorWorker(db_manager, vector_store, embedder) # Assuming VectorWorker exists
        emotion_worker = EmotionWorker(db_manager, emotion_manager)
        # parser_worker = ParserWorker(...) # Assuming ParserWorker exists
        graph_worker = GraphWorker(graph_manager)

        # --- Worker Manager Setup ---
        manager = WorkerManager()

        # Register workers
        manager.register_worker("conversation", conv_worker)
        manager.register_worker("database", db_worker)
        # manager.register_worker("vector", vector_worker)
        manager.register_worker("emotion", emotion_worker)
        # manager.register_worker("parser", parser_worker)
        manager.register_worker("graph", graph_worker)

        print("Starting Worker Manager...")
        with manager:  # Use context manager to start/stop
            print("Worker Manager started. Running for 15 seconds...")
            time.sleep(5)

            print("\n--- Current Status ---")
            status = manager.get_status()
            for name, stat in status.items():
                print(
                    f"Worker '{name}': State={stat.get('state', 'unknown')}, Running={stat.get('running', False)}"
                )

            print("\n--- Current Metrics ---")
            metrics = manager.get_metrics()
            for name, met in metrics.items():
                # Check if metrics are supported before printing details
                if met.get("status") == "Metrics not supported":
                    print(f"Worker '{name}': Metrics not supported")
                elif met.get("error"):
                    print(
                        f"Worker '{name}': Error retrieving metrics - {met.get('error')}"
                    )
                else:
                    # Print available metrics, handling potential missing keys gracefully
                    op_count = met.get("operation_count", "N/A")
                    err_count = met.get("error_count", "N/A")
                    print(f"Worker '{name}': Ops={op_count}, Errors={err_count}")

            print("\nPausing workers for 3 seconds...")
            manager.pause_all()
            time.sleep(3)
            print("Resuming workers...")
            manager.resume_all()
            time.sleep(7)  # Run for another 7 seconds

        print("\nWorker Manager stopped.")

        # Add a failsafe timeout for status retrieval
        try:

            with threading.RLock():
                # Prevent hanging on status retrieval
                timer = threading.Timer(5.0, lambda: _thread.interrupt_main())
                timer.daemon = True
                timer.start()

                try:
                    final_status = manager.get_status()
                    print("\n--- Final Status ---")
                    for name, stat in final_status.items():
                        print(
                            f"Worker '{name}': State={stat.get('state', 'unknown')}, "
                            f"Running={stat.get('running', False)}"
                        )
                finally:
                    timer.cancel()
        except KeyboardInterrupt:
            print("\n--- Final status retrieval timed out ---")

    except ImportError as e:
        print(f"\nDEMO SKIPPED: Missing dependency - {e}")
        print("Please ensure all required Word Forge components are installed.")
    except Exception as e:
        print(f"\nAn error occurred during the demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    worker_manager_demo()
