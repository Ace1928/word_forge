import sqlite3
from typing import Any, Dict, Optional

from word_forge.database.db_manager import DBManager


class EmotionManager:
    """
    Tracks emotional associations with words (or conversation messages).
    E.g. storing valence (positive/negative), arousal, etc.
    """

    def __init__(self, db_manager: DBManager):
        self.db_manager = db_manager
        self._create_table()

    def _create_table(self):
        """
        We'll create a 'word_emotion' table that links word_id to emotional values,
        plus an optional 'message_emotion' table for conversation messages.
        """
        with sqlite3.connect(self.db_manager.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS word_emotion (
                    word_id INTEGER NOT NULL,
                    valence REAL DEFAULT 0.0,   -- -1.0 (very negative) to +1.0 (very positive)
                    arousal REAL DEFAULT 0.0,   -- 0.0 (calm) to 1.0 (excited)
                    last_updated REAL DEFAULT (strftime('%s','now')),
                    UNIQUE(word_id) ON CONFLICT REPLACE,
                    FOREIGN KEY(word_id) REFERENCES words(id)
                )
            """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS message_emotion (
                    message_id INTEGER NOT NULL,
                    emotion_label TEXT,
                    confidence REAL DEFAULT 1.0,
                    last_updated REAL DEFAULT (strftime('%s','now')),
                    UNIQUE(message_id) ON CONFLICT REPLACE
                )
            """
            )
            conn.commit()

    def set_word_emotion(self, word_id: int, valence: float, arousal: float):
        """
        Store or update emotional values for a given word ID.
        """
        with sqlite3.connect(self.db_manager.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO word_emotion (word_id, valence, arousal)
                VALUES (?, ?, ?)
                ON CONFLICT(word_id)
                DO UPDATE SET valence=excluded.valence,
                              arousal=excluded.arousal,
                              last_updated=strftime('%s','now')
            """,
                (word_id, valence, arousal),
            )
            conn.commit()

    def get_word_emotion(self, word_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve emotional data for a word.
        """
        with sqlite3.connect(self.db_manager.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                SELECT valence, arousal, last_updated
                FROM word_emotion
                WHERE word_id = ?
            """,
                (word_id,),
            )
            row = c.fetchone()
            if row:
                return {
                    "valence": row[0],
                    "arousal": row[1],
                    "last_updated": row[2],
                }
            return None

    def set_message_emotion(self, message_id: int, label: str, confidence: float = 1.0):
        """
        Tag a conversation message with an emotion label (e.g. 'happy', 'sad').
        """
        with sqlite3.connect(self.db_manager.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO message_emotion (message_id, emotion_label, confidence)
                VALUES (?, ?, ?)
                ON CONFLICT(message_id)
                DO UPDATE SET emotion_label=excluded.emotion_label,
                              confidence=excluded.confidence,
                              last_updated=strftime('%s','now')
            """,
                (message_id, label, confidence),
            )
            conn.commit()

    def get_message_emotion(self, message_id: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_manager.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                SELECT emotion_label, confidence, last_updated
                FROM message_emotion
                WHERE message_id = ?
            """,
                (message_id,),
            )
            row = c.fetchone()
            if row:
                return {
                    "emotion_label": row[0],
                    "confidence": row[1],
                    "last_updated": row[2],
                }
            return None
