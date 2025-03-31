import sqlite3
from typing import Any, Dict, Optional

from word_forge.database.db_manager import DBManager


class ConversationManager:
    """
    Manages conversation sessions and messages.
    Stores data in new tables: 'conversations' and 'conversation_messages'.
    """

    def __init__(self, db_manager: DBManager):
        self.db_manager = db_manager
        self._create_tables()

    def _create_tables(self):
        """
        Create tables if they don't already exist:
          conversations(id, status, created_at, updated_at)
          conversation_messages(id, conversation_id, speaker, text, timestamp)
        """
        with sqlite3.connect(self.db_manager.db_path) as conn:
            c = conn.cursor()
            # Table for the overarching conversation sessions
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT DEFAULT 'ACTIVE',
                    created_at REAL DEFAULT (strftime('%s','now')),
                    updated_at REAL DEFAULT (strftime('%s','now'))
                )
            """
            )
            # Table for individual messages in a conversation
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    speaker TEXT,
                    text TEXT,
                    timestamp REAL DEFAULT (strftime('%s','now')),
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                )
            """
            )
            conn.commit()

    def start_conversation(self) -> int:
        """
        Creates a new conversation row and returns its ID.
        """
        with sqlite3.connect(self.db_manager.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO conversations (status) VALUES ('ACTIVE')
            """
            )
            conversation_id = c.lastrowid
            conn.commit()
        return conversation_id

    def end_conversation(self, conversation_id: int):
        """
        Marks a conversation as COMPLETED.
        """
        with sqlite3.connect(self.db_manager.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                UPDATE conversations
                SET status = 'COMPLETED',
                    updated_at = strftime('%s','now')
                WHERE id = ?
            """,
                (conversation_id,),
            )
            conn.commit()

    def add_message(self, conversation_id: int, speaker: str, text: str):
        """
        Appends a message to the specified conversation.
        """
        with sqlite3.connect(self.db_manager.db_path) as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO conversation_messages (conversation_id, speaker, text)
                VALUES (?, ?, ?)
            """,
                (conversation_id, speaker, text),
            )
            # Also update updated_at in the conversation
            c.execute(
                """
                UPDATE conversations
                SET updated_at = strftime('%s','now')
                WHERE id = ?
            """,
                (conversation_id,),
            )
            conn.commit()

    def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """
        Returns conversation info, including messages, if found.
        """
        with sqlite3.connect(self.db_manager.db_path) as conn:
            c = conn.cursor()
            # Fetch conversation row
            c.execute(
                """
                SELECT id, status, created_at, updated_at
                FROM conversations
                WHERE id = ?
            """,
                (conversation_id,),
            )
            row = c.fetchone()
            if not row:
                return None

            conv_data = {
                "id": row[0],
                "status": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "messages": [],
            }

            # Fetch messages
            c.execute(
                """
                SELECT id, speaker, text, timestamp
                FROM conversation_messages
                WHERE conversation_id = ?
                ORDER BY id ASC
            """,
                (conversation_id,),
            )
            messages = c.fetchall()
            for m in messages:
                conv_data["messages"].append(
                    {
                        "id": m[0],
                        "speaker": m[1],
                        "text": m[2],
                        "timestamp": m[3],
                    }
                )
            return conv_data
