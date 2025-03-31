import random
import sqlite3
import threading
import time
import traceback
from dataclasses import dataclass
from typing import List

from word_forge.database.db_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager


@dataclass
class WordEmotion:
    """Data structure representing a word needing emotional classification."""

    word_id: int
    term: str


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
        poll_interval: float = 5.0,
        batch_size: int = 10,
        daemon: bool = True,
    ):
        """
        Initialize the emotion processing worker thread.

        Args:
            db: Database manager for accessing word data
            emotion_manager: Manager for storing emotional attributes
            poll_interval: Seconds to wait between processing cycles
            batch_size: Maximum number of words to process per cycle
            daemon: Whether thread should run as daemon
        """
        super().__init__(daemon=daemon)
        self.db = db
        self.emotion_manager = emotion_manager
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self._stop_flag = False

    def run(self) -> None:
        """
        Main worker loop that processes words without emotional attributes.

        Continuously polls database for unprocessed words, assigns them
        emotional values, and handles any exceptions during processing.
        """
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
                time.sleep(self.poll_interval)

    def stop(self) -> None:
        """Signal the worker thread to stop gracefully."""
        self._stop_flag = True

    def _get_unemotioned_words(self) -> List[WordEmotion]:
        """
        Retrieve words from the database that lack emotional attributes.

        Returns:
            List of WordEmotion objects representing unprocessed words
        """
        query = """
        SELECT w.id, w.term
        FROM words w
        LEFT JOIN word_emotion we ON w.id = we.word_id
        WHERE we.word_id IS NULL
        LIMIT ?
        """
        connection = sqlite3.connect(self.db.db_path)
        try:
            cursor = connection.cursor()
            cursor.execute(query, (self.batch_size,))
            rows = cursor.fetchall()
            return [WordEmotion(word_id=row[0], term=row[1]) for row in rows]
        finally:
            connection.close()

    def _process_word_emotions(self, words: List[WordEmotion]) -> None:
        """
        Assign emotional values to each word and store in the database.

        Args:
            words: List of words to process
        """
        for word in words:
            valence = random.uniform(-1.0, 1.0)
            arousal = random.uniform(0.0, 1.0)
            self.emotion_manager.set_word_emotion(word.word_id, valence, arousal)
            print(f"Tagged {word.term}: valence={valence:.2f}, arousal={arousal:.2f}")

    def _handle_processing_error(self, error: Exception) -> None:
        """
        Log processing errors with contextual information.

        Args:
            error: The exception that occurred
        """
        print(f"[EmotionWorker] Error: {str(error)}")
        traceback.print_exc()


def main() -> None:
    """Demonstrate EmotionWorker initialization and operation."""
    from word_forge.database.db_manager import DBManager
    from word_forge.emotion.emotion_manager import EmotionManager

    # Initialize dependencies with a real file path
    db_path = "word_forge.sqlite"
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
                word_id = db._get_word_id(word)
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
