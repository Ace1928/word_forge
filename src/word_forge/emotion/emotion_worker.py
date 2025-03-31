import random
import threading
import time
import traceback

from word_forge.database.db_manager import DBManager
from word_forge.emotion.emotion_manager import EmotionManager


class EmotionWorker(threading.Thread):
    """
    Continuously scans new words or messages to assign emotional values or tags.
    Could integrate an external sentiment/emotion classifier.
    """

    def __init__(
        self,
        db: DBManager,
        emotion_manager: EmotionManager,
        poll_interval=5.0,
        daemon=True,
    ):
        super().__init__(daemon=daemon)
        self.db = db
        self.emotion_manager = emotion_manager
        self.poll_interval = poll_interval
        self._stop_flag = False

    def run(self):
        while not self._stop_flag:
            try:
                # Example: find words that do not have an entry in word_emotion yet
                words_to_tag = self._get_unemotioned_words()
                for w_id, term in words_to_tag:
                    # Dummy approach: random valence/arousal
                    valence = random.uniform(-1.0, 1.0)
                    arousal = random.uniform(0.0, 1.0)
                    self.emotion_manager.set_word_emotion(w_id, valence, arousal)
                # Sleep until next poll
                time.sleep(self.poll_interval)
            except Exception as e:
                print(f"[EmotionWorker] Error: {str(e)}")
                traceback.print_exc()
                time.sleep(self.poll_interval)

    def stop(self):
        self._stop_flag = True

    def _get_unemotioned_words(self):
        """
        Return a list of (id, term) for words that don't appear in word_emotion.
        """
        query = """
        SELECT w.id, w.term
        FROM words w
        LEFT JOIN word_emotion we ON w.id = we.word_id
        WHERE we.word_id IS NULL
        LIMIT 10
        """
        with self.db._create_connection() as conn:
            c = conn.cursor()
            c.execute(query)
            rows = c.fetchall()
        return [(r[0], r[1]) for r in rows]
