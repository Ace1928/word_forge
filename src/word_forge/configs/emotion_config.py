"""
Emotional Configuration module for Word Forge.
"""

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List

from word_forge.configs.config_essentials import EmotionRange
from word_forge.configs.config_types import EnvMapping

# SQL template constants exported at module level for backward compatibility
SQL_CREATE_WORD_EMOTION_TABLE = """
    CREATE TABLE IF NOT EXISTS word_emotion (
        word_id INTEGER PRIMARY KEY,
        valence REAL NOT NULL,
        arousal REAL NOT NULL,
        timestamp REAL NOT NULL,
        FOREIGN KEY(word_id) REFERENCES words(id)
    );
"""

SQL_CREATE_MESSAGE_EMOTION_TABLE = """
    CREATE TABLE IF NOT EXISTS message_emotion (
        message_id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp REAL NOT NULL
    );
"""

SQL_INSERT_WORD_EMOTION = """
    INSERT OR REPLACE INTO word_emotion
    (word_id, valence, arousal, timestamp)
    VALUES (?, ?, ?, ?)
"""

SQL_GET_WORD_EMOTION = """
    SELECT word_id, valence, arousal, timestamp
    FROM word_emotion
    WHERE word_id = ?
"""

SQL_INSERT_MESSAGE_EMOTION = """
    INSERT OR REPLACE INTO message_emotion
    (message_id, label, confidence, timestamp)
    VALUES (?, ?, ?, ?)
"""

SQL_GET_MESSAGE_EMOTION = """
    SELECT message_id, label, confidence, timestamp
    FROM message_emotion
    WHERE message_id = ?
"""


@dataclass
class EmotionConfig:
    """
    Configuration for emotion analysis.

    Controls sentiment analysis parameters, emotion classification rules,
    and database schema for emotion data.

    Attributes:
        enable_vader: Whether to use VADER for sentiment analysis
        vader_weight: Weight given to VADER in hybrid sentiment analysis
        textblob_weight: Weight given to TextBlob in hybrid sentiment analysis
        valence_range: Range constraints for valence values
        arousal_range: Range constraints for arousal values
        confidence_range: Range constraints for confidence levels
        sql_templates: SQL templates for emotion tables and queries
        emotion_keywords: Emotion category keywords for classification
        min_keyword_confidence: Minimum confidence when no keywords found
        keyword_match_weight: Weight given to keyword matches in classification
        ENV_VARS: Mapping of environment variables to config attributes
    """

    # VADER configuration
    enable_vader: bool = True

    # Default mixing weights for hybrid sentiment analysis
    vader_weight: float = 0.7
    textblob_weight: float = 0.3

    # Emotion range constraints
    valence_range: EmotionRange = (-1.0, 1.0)  # Negative to positive
    arousal_range: EmotionRange = (0.0, 1.0)  # Calm to excited
    confidence_range: EmotionRange = (0.0, 1.0)  # Certainty level

    # SQL templates for emotion tables
    sql_templates: Dict[str, str] = field(
        default_factory=lambda: {
            "create_word_emotion_table": SQL_CREATE_WORD_EMOTION_TABLE,
            "create_message_emotion_table": SQL_CREATE_MESSAGE_EMOTION_TABLE,
            "insert_word_emotion": SQL_INSERT_WORD_EMOTION,
            "get_word_emotion": SQL_GET_WORD_EMOTION,
            "insert_message_emotion": SQL_INSERT_MESSAGE_EMOTION,
            "get_message_emotion": SQL_GET_MESSAGE_EMOTION,
        }
    )

    # Emotion category keywords for classification
    emotion_keywords: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "happiness": ["happy", "joy", "delight", "pleased", "glad", "excited"],
            "sadness": ["sad", "unhappy", "depressed", "down", "miserable", "gloomy"],
            "anger": ["angry", "furious", "enraged", "mad", "irritated", "annoyed"],
            "fear": [
                "afraid",
                "scared",
                "frightened",
                "terrified",
                "anxious",
                "worried",
            ],
            "surprise": ["surprised", "astonished", "amazed", "shocked", "startled"],
            "disgust": ["disgusted", "revolted", "repulsed", "sickened", "appalled"],
            "neutral": ["okay", "fine", "neutral", "indifferent", "average"],
        }
    )

    # Analysis parameters
    min_keyword_confidence: float = 0.3  # Minimum confidence when no keywords found
    keyword_match_weight: float = (
        0.6  # Weight given to keyword matches in classification
    )

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_ENABLE_VADER": ("enable_vader", bool),
    }

    def is_valid_valence(self, value: float) -> bool:
        """
        Check if a valence value is within the configured range.

        Args:
            value: The valence value to validate

        Returns:
            True if the value is within the valid range
        """
        min_val, max_val = self.valence_range
        return min_val <= value <= max_val

    def is_valid_arousal(self, value: float) -> bool:
        """
        Check if an arousal value is within the configured range.

        Args:
            value: The arousal value to validate

        Returns:
            True if the value is within the valid range
        """
        min_val, max_val = self.arousal_range
        return min_val <= value <= max_val

    def is_valid_confidence(self, value: float) -> bool:
        """
        Check if a confidence value is within the configured range.

        Args:
            value: The confidence value to validate

        Returns:
            True if the value is within the valid range
        """
        min_val, max_val = self.confidence_range
        return min_val <= value <= max_val

    # For backward compatibility, maintain access to SQL templates through class attributes
    @property
    def SQL_CREATE_WORD_EMOTION_TABLE(self) -> str:
        """SQL template for word emotion table creation."""
        return self.sql_templates["create_word_emotion_table"]

    @property
    def SQL_CREATE_MESSAGE_EMOTION_TABLE(self) -> str:
        """SQL template for message emotion table creation."""
        return self.sql_templates["create_message_emotion_table"]

    @property
    def SQL_INSERT_WORD_EMOTION(self) -> str:
        """SQL template for inserting word emotion data."""
        return self.sql_templates["insert_word_emotion"]

    @property
    def SQL_GET_WORD_EMOTION(self) -> str:
        """SQL template for retrieving word emotion data."""
        return self.sql_templates["get_word_emotion"]

    @property
    def SQL_INSERT_MESSAGE_EMOTION(self) -> str:
        """SQL template for inserting message emotion data."""
        return self.sql_templates["insert_message_emotion"]

    @property
    def SQL_GET_MESSAGE_EMOTION(self) -> str:
        """SQL template for retrieving message emotion data."""
        return self.sql_templates["get_message_emotion"]
