import random
import sqlite3
import threading
import time
import traceback
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Protocol, TypedDict, final

from word_forge.config import config
from word_forge.database.db_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager


class EmotionState(Enum):
    """Worker lifecycle states for monitoring and control."""

    RUNNING = auto()
    STOPPED = auto()
    ERROR = auto()

    def __str__(self) -> str:
        """Return lowercase state name for consistent string representation."""
        return self.name.lower()


class EmotionError(Exception):
    """Base exception for emotion processing errors."""

    pass


class EmotionDBError(EmotionError):
    """Raised when database operations for emotion processing fail."""

    pass


class EmotionProcessingError(EmotionError):
    """Raised when word emotion classification fails."""

    pass


@dataclass(frozen=True)
class WordEmotion:
    """Immutable data structure representing a word needing emotional classification."""

    word_id: int
    term: str


class EmotionWorkerStatus(TypedDict):
    """Type definition for worker status information."""

    running: bool
    processed_count: int
    error_count: int
    last_update: Optional[float]
    uptime: Optional[float]
    state: str


class EmotionWorkerInterface(Protocol):
    """Protocol defining the required interface for emotion workers."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_status(self) -> EmotionWorkerStatus: ...
    def is_alive(self) -> bool: ...


@final
class EmotionWorker(threading.Thread):
    """
    Asynchronously assigns emotional values to unprocessed words in the database.

    This worker continuously polls the database for words without emotional
    attributes, then applies valence and arousal values to these words using
    the EmotionManager.
    """

    def __init__(
        self,
        db: DBManager,
        emotion_manager: EmotionManager,
        poll_interval: Optional[float] = None,
        batch_size: Optional[int] = None,
        daemon: bool = True,
    ):
        """
        Initialize the emotion processing worker thread.

        Args:
            db: Database manager for accessing word data
            emotion_manager: Manager for storing emotional attributes
            poll_interval: Seconds to wait between processing cycles (defaults to config)
            batch_size: Maximum number of words to process per cycle (defaults to config)
            daemon: Whether thread should run as daemon
        """
        super().__init__(daemon=daemon)
        self.db = db
        self.emotion_manager = emotion_manager
        self.poll_interval = poll_interval or 5.0
        self.batch_size = batch_size or config.queue.batch_size

        self._stop_flag = False
        self._current_state = EmotionState.STOPPED
        self._status_lock = threading.RLock()
        self._start_time: Optional[float] = None
        self._last_update: Optional[float] = None
        self._processed_count = 0
        self._error_count = 0

    def run(self) -> None:
        """
        Main worker loop that processes words without emotional attributes.

        Continuously polls database for unprocessed words, assigns them
        emotional values, and handles any exceptions during processing.
        """
        with self._status_lock:
            self._start_time = time.time()
            self._current_state = EmotionState.RUNNING

        while not self._stop_flag:
            try:
                words_to_tag = self._get_unemotioned_words()
                if words_to_tag:
                    print(f"Processing {len(words_to_tag)} words...")
                    self._process_word_emotions(words_to_tag)
                else:
                    print("No words to process, waiting...")
                time.sleep(self.poll_interval)
            except Exception as e:
                self._handle_processing_error(e)
                time.sleep(max(1.0, self.poll_interval / 2))

        with self._status_lock:
            self._current_state = EmotionState.STOPPED

    def stop(self) -> None:
        """Signal the worker thread to stop gracefully."""
        self._stop_flag = True

    def get_status(self) -> EmotionWorkerStatus:
        """
        Return the current status of the emotion worker.

        Returns:
            Dictionary containing operational metrics including:
            - running: Whether the worker is active
            - processed_count: Number of successfully processed words
            - error_count: Number of encountered errors
            - last_update: Timestamp of last successful update
            - uptime: Seconds since thread start if running
            - state: Current worker state as string
        """
        with self._status_lock:
            uptime = None
            if self._start_time:
                uptime = time.time() - self._start_time

            status: EmotionWorkerStatus = {
                "running": self.is_alive() and not self._stop_flag,
                "processed_count": self._processed_count,
                "error_count": self._error_count,
                "last_update": self._last_update,
                "uptime": uptime,
                "state": str(self._current_state),
            }

            return status

    def _get_unemotioned_words(self) -> List[WordEmotion]:
        """
        Retrieve words from the database that lack emotional attributes.

        Returns:
            List of WordEmotion objects representing unprocessed words

        Raises:
            EmotionDBError: If database access fails
        """
        query = """
        SELECT w.id, w.term
        FROM words w
        LEFT JOIN word_emotion we ON w.id = we.word_id
        WHERE we.word_id IS NULL
        LIMIT ?
        """
        connection = None
        try:
            connection = sqlite3.connect(self.db.db_path)
            cursor = connection.cursor()
            cursor.execute(query, (self.batch_size,))
            rows = cursor.fetchall()
            return [WordEmotion(word_id=row[0], term=row[1]) for row in rows]
        except sqlite3.Error as e:
            raise EmotionDBError(f"Failed to retrieve words: {str(e)}") from e
        finally:
            if connection is not None:
                connection.close()

    def _process_word_emotions(self, words: List[WordEmotion]) -> None:
        """
        Assign emotional values to each word and store in the database.

        Args:
            words: List of words to process

        Raises:
            EmotionProcessingError: If emotion assignment fails
        """
        for word in words:
            try:
                # Get the emotional valence/arousal ranges from config
                valence_range = config.emotion.valence_range
                arousal_range = config.emotion.arousal_range

                # Generate values within the configured ranges
                valence = random.uniform(valence_range[0], valence_range[1])
                arousal = random.uniform(arousal_range[0], arousal_range[1])

                self.emotion_manager.set_word_emotion(word.word_id, valence, arousal)

                with self._status_lock:
                    self._processed_count += 1
                    self._last_update = time.time()

                print(
                    f"Tagged {word.term}: valence={valence:.2f}, arousal={arousal:.2f}"
                )
            except Exception as e:
                raise EmotionProcessingError(
                    f"Failed to process word {word.term}: {str(e)}"
                ) from e

    def _handle_processing_error(self, error: Exception) -> None:
        """
        Log processing errors with contextual information and update state.

        Args:
            error: The exception that occurred
        """
        with self._status_lock:
            self._error_count += 1
            self._current_state = EmotionState.ERROR

        error_type = type(error).__name__
        print(f"[EmotionWorker] {error_type}: {str(error)}")
        traceback.print_exc()


def main() -> None:
    """Demonstrate EmotionWorker initialization and operation."""
    from word_forge.database.db_manager import DBManager
    from word_forge.emotion.emotion_manager import EmotionManager

    # Initialize dependencies with a configured file path
    db_path = config.database.db_path
    db = DBManager(db_path)
    emotion_manager = EmotionManager(db)

    # Seed the database with some sample words if needed
    sample_words = [
        "happiness",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "trust",
        "anticipation",
    ]
    print("Ensuring sample words exist in database...")
    for word in sample_words:
        try:
            db.insert_or_update_word(word, f"The emotion of {word}", "noun")
            print(f"Added/updated word: {word}")
        except Exception as e:
            print(f"Error adding {word}: {e}")

    # Configure and start the worker
    worker = EmotionWorker(
        db=db, emotion_manager=emotion_manager, poll_interval=10.0, batch_size=20
    )

    try:
        print("Starting emotion worker...")
        worker.start()

        # Let it run for a while
        print("Worker will run for 60 seconds...")
        time.sleep(60)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received")
    finally:
        print("Stopping emotion worker...")
        worker.stop()
        worker.join(timeout=5.0)
        print("Worker stopped.")

        # Display the emotional values assigned to words
        print("\nEmotional values assigned to words:")
        print("-" * 60)
        for word in sample_words:
            try:
                word_id = db.get_word_id(word)
                if word_id:
                    emotion_data = emotion_manager.get_word_emotion(word_id)
                    if emotion_data:
                        print(
                            f"{word}: valence={emotion_data['valence']:.2f}, "
                            f"arousal={emotion_data['arousal']:.2f}"
                        )
                    else:
                        print(f"{word}: No emotional data assigned")
            except Exception as e:
                print(f"Error retrieving emotion for {word}: {e}")


if __name__ == "__main__":
    main()
