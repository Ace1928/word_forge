"""
Emotional Configuration module for Word Forge.

This module provides configuration parameters for emotion analysis operations,
including sentiment analysis settings, emotional classification rules, and
database schema definitions for storing emotion-related data.

The module contains:
    - SQL template constants for database operations
    - EmotionConfig dataclass with configurable parameters
    - Validation methods for emotional measurements
    - Property accessors for backward compatibility

Architecture:
    ┌──────────────────────┐
    │   EmotionConfig      │
    └──────────┬───────────┘
               │
    ┌──────────┴───────────┐
    │  Emotion Parameters  │
    └──────────────────────┘
    ┌─────┬─────┬──────────┐
    │ SQL │Range│Classif.  │
    └─────┴─────┴──────────┘
"""

from dataclasses import dataclass, field
from typing import (
    ClassVar,
    Dict,
    Final,
    List,
    Literal,
    NamedTuple,
    Protocol,
    TypedDict,
    TypeVar,
)

from word_forge.configs.config_essentials import EmotionRange, EnvMapping

# Type definitions for better structural typing
EmotionCategory = Literal[
    "happiness", "sadness", "anger", "fear", "surprise", "disgust", "neutral"
]

T = TypeVar("T")  # Generic type for validation results


class EmotionValidationResult(NamedTuple):
    """Result of emotion value validation with detailed context."""

    is_valid: bool
    """Whether the value is valid according to configuration constraints."""

    message: str
    """Descriptive message explaining validation result."""

    value: float
    """The value that was validated."""

    range: EmotionRange
    """The range against which the value was validated."""


class SQLTemplateProtocol(Protocol):
    """Protocol defining the SQL template interface for emotion operations."""

    @property
    def create_word_emotion_table(self) -> str: ...

    @property
    def create_message_emotion_table(self) -> str: ...

    @property
    def insert_word_emotion(self) -> str: ...

    @property
    def get_word_emotion(self) -> str: ...

    @property
    def insert_message_emotion(self) -> str: ...

    @property
    def get_message_emotion(self) -> str: ...


class EmotionKeywordsDict(TypedDict):
    """Type definition for the emotion keywords dictionary."""

    happiness: List[str]
    sadness: List[str]
    anger: List[str]
    fear: List[str]
    surprise: List[str]
    disgust: List[str]
    neutral: List[str]


# Define a TypedDict for VADER sentiment scores
class VaderSentimentScores(TypedDict):
    """Type definition for VADER sentiment analyzer output."""

    pos: float
    neg: float
    neu: float
    compound: float


# ==========================================
# SQL Template Constants
# ==========================================
# Exported at module level for backward compatibility

#: SQL schema for word emotion table
SQL_CREATE_WORD_EMOTION_TABLE: Final[
    str
] = """
    CREATE TABLE IF NOT EXISTS word_emotion (
        word_id INTEGER PRIMARY KEY,
        valence REAL NOT NULL,
        arousal REAL NOT NULL,
        timestamp REAL NOT NULL,
        FOREIGN KEY(word_id) REFERENCES words(id)
    );
"""

#: SQL schema for message emotion table
SQL_CREATE_MESSAGE_EMOTION_TABLE: Final[
    str
] = """
    CREATE TABLE IF NOT EXISTS message_emotion (
        message_id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        confidence REAL NOT NULL,
        timestamp REAL NOT NULL
    );
"""

#: SQL query for inserting word emotion data
SQL_INSERT_WORD_EMOTION: Final[
    str
] = """
    INSERT OR REPLACE INTO word_emotion
    (word_id, valence, arousal, timestamp)
    VALUES (?, ?, ?, ?)
"""

#: SQL query for retrieving word emotion data
SQL_GET_WORD_EMOTION: Final[
    str
] = """
    SELECT word_id, valence, arousal, timestamp
    FROM word_emotion
    WHERE word_id = ?
"""

#: SQL query for inserting message emotion data
SQL_INSERT_MESSAGE_EMOTION: Final[
    str
] = """
    INSERT OR REPLACE INTO message_emotion
    (message_id, label, confidence, timestamp)
    VALUES (?, ?, ?, ?)
"""

#: SQL query for retrieving message emotion data
SQL_GET_MESSAGE_EMOTION: Final[
    str
] = """
    SELECT message_id, label, confidence, timestamp
    FROM message_emotion
    WHERE message_id = ?
"""


@dataclass
class EmotionConfig:
    """
    Configuration for emotion analysis operations.

    Controls sentiment analysis parameters, emotion classification rules,
    and database schema for emotion data storage and retrieval.

    Attributes:
        enable_vader: Whether to use VADER for sentiment analysis
        vader_weight: Weight given to VADER in hybrid sentiment analysis
        textblob_weight: Weight given to TextBlob in hybrid sentiment analysis
        valence_range: Valid range for valence values (negative to positive)
        arousal_range: Valid range for arousal values (calm to excited)
        confidence_range: Valid range for confidence levels (certainty)
        sql_templates: Dictionary of SQL templates for emotion data operations
        emotion_keywords: Dictionary mapping emotion categories to keywords
        min_keyword_confidence: Minimum confidence when no keywords found
        keyword_match_weight: Weight given to keyword matches in classification
        ENV_VARS: Mapping of environment variables to config attributes

    Examples:
        >>> config = EmotionConfig()
        >>> config.is_valid_valence(0.5)
        True
        >>> config.is_valid_valence(2.0)  # Out of range
        False
        >>> config.SQL_GET_WORD_EMOTION  # Access SQL via property
        'SELECT word_id, valence, arousal, timestamp\\n    FROM word_emotion\\n    WHERE word_id = ?'
    """

    # ==========================================
    # Sentiment Analysis Configuration
    # ==========================================

    #: Whether to use VADER for sentiment analysis
    enable_vader: bool = True

    #: Weight given to VADER in hybrid sentiment analysis
    vader_weight: float = 0.7

    #: Weight given to TextBlob in hybrid sentiment analysis
    textblob_weight: float = 0.3

    # ==========================================
    # Emotion Parameter Constraints
    # ==========================================

    #: Valid range for valence values (negative to positive)
    valence_range: EmotionRange = (-1.0, 1.0)

    #: Valid range for arousal values (calm to excited)
    arousal_range: EmotionRange = (0.0, 1.0)

    #: Valid range for confidence values (certainty level)
    confidence_range: EmotionRange = (0.0, 1.0)

    # ==========================================
    # SQL Templates Configuration
    # ==========================================

    #: SQL templates for emotion database operations
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

    # ==========================================
    # Emotion Classification Configuration
    # ==========================================

    #: Dictionary mapping emotion categories to keywords
    emotion_keywords: EmotionKeywordsDict = field(
        default_factory=lambda: {
            "happiness": ["happy", "joy", "delight", "pleased", "glad", "excited"],
            "sadness": [
                "sad",
                "unhappy",
                "depressed",
                "down",
                "miserable",
                "gloomy",
            ],
            "anger": ["angry", "furious", "enraged", "mad", "irritated", "annoyed"],
            "fear": [
                "afraid",
                "scared",
                "frightened",
                "terrified",
                "anxious",
                "worried",
            ],
            "surprise": [
                "surprised",
                "astonished",
                "amazed",
                "shocked",
                "startled",
            ],
            "disgust": [
                "disgusted",
                "revolted",
                "repulsed",
                "sickened",
                "appalled",
            ],
            "neutral": ["okay", "fine", "neutral", "indifferent", "average"],
        }
    )

    #: Minimum confidence value when no keywords are found
    min_keyword_confidence: float = 0.3

    #: Weight given to keyword matches in emotion classification
    keyword_match_weight: float = 0.6

    # ==========================================
    # Environment Variable Configuration
    # ==========================================

    #: Mapping of environment variables to configuration attributes
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_ENABLE_VADER": ("enable_vader", bool),
        "WORD_FORGE_VADER_WEIGHT": ("vader_weight", float),
        "WORD_FORGE_TEXTBLOB_WEIGHT": ("textblob_weight", float),
        "WORD_FORGE_MIN_KEYWORD_CONFIDENCE": ("min_keyword_confidence", float),
        "WORD_FORGE_KEYWORD_MATCH_WEIGHT": ("keyword_match_weight", float),
    }

    # ==========================================
    # Validation Methods
    # ==========================================

    def is_valid_valence(self, value: float) -> bool:
        """
        Check if a valence value is within the configured range.

        Valence represents the positive or negative quality of an emotion,
        typically ranging from -1.0 (highly negative) to 1.0 (highly positive).

        Args:
            value: The valence value to validate

        Returns:
            bool: True if the value is within the valid range, False otherwise

        Examples:
            >>> config = EmotionConfig()
            >>> config.is_valid_valence(0.7)
            True
            >>> config.is_valid_valence(-1.5)
            False
        """
        min_val, max_val = self.valence_range
        return min_val <= value <= max_val

    def validate_valence(self, value: float) -> EmotionValidationResult:
        """
        Validate a valence value with detailed contextual information.

        Provides a rich validation result containing not just a boolean indicator,
        but also descriptive messages and context about the validation.

        Args:
            value: The valence value to validate

        Returns:
            EmotionValidationResult: Structured validation result with context

        Examples:
            >>> config = EmotionConfig()
            >>> result = config.validate_valence(0.7)
            >>> result.is_valid
            True
            >>> result.message
            'Valence value 0.7 is within valid range (-1.0, 1.0)'

            >>> result = config.validate_valence(-1.5)
            >>> result.is_valid
            False
            >>> result.message
            'Valence value -1.5 is outside valid range (-1.0, 1.0)'
        """
        min_val, max_val = self.valence_range
        is_valid = min_val <= value <= max_val

        if is_valid:
            message = (
                f"Valence value {value} is within valid range ({min_val}, {max_val})"
            )
        else:
            message = (
                f"Valence value {value} is outside valid range ({min_val}, {max_val})"
            )

        return EmotionValidationResult(
            is_valid=is_valid, message=message, value=value, range=self.valence_range
        )

    def is_valid_arousal(self, value: float) -> bool:
        """
        Check if an arousal value is within the configured range.

        Arousal represents the intensity or activation level of an emotion,
        typically ranging from 0.0 (calm) to 1.0 (excited).

        Args:
            value: The arousal value to validate

        Returns:
            bool: True if the value is within the valid range, False otherwise

        Examples:
            >>> config = EmotionConfig()
            >>> config.is_valid_arousal(0.5)
            True
            >>> config.is_valid_arousal(1.2)
            False
        """
        min_val, max_val = self.arousal_range
        return min_val <= value <= max_val

    def validate_arousal(self, value: float) -> EmotionValidationResult:
        """
        Validate an arousal value with detailed contextual information.

        Provides a rich validation result containing not just a boolean indicator,
        but also descriptive messages and context about the validation.

        Args:
            value: The arousal value to validate

        Returns:
            EmotionValidationResult: Structured validation result with context

        Examples:
            >>> config = EmotionConfig()
            >>> result = config.validate_arousal(0.5)
            >>> result.is_valid
            True
            >>> result.message
            'Arousal value 0.5 is within valid range (0.0, 1.0)'

            >>> result = config.validate_arousal(1.2)
            >>> result.is_valid
            False
            >>> result.message
            'Arousal value 1.2 is outside valid range (0.0, 1.0)'
        """
        min_val, max_val = self.arousal_range
        is_valid = min_val <= value <= max_val

        if is_valid:
            message = (
                f"Arousal value {value} is within valid range ({min_val}, {max_val})"
            )
        else:
            message = (
                f"Arousal value {value} is outside valid range ({min_val}, {max_val})"
            )

        return EmotionValidationResult(
            is_valid=is_valid, message=message, value=value, range=self.arousal_range
        )

    def is_valid_confidence(self, value: float) -> bool:
        """
        Check if a confidence value is within the configured range.

        Confidence represents the certainty level in an emotion classification,
        typically ranging from 0.0 (complete uncertainty) to 1.0 (complete certainty).

        Args:
            value: The confidence value to validate

        Returns:
            bool: True if the value is within the valid range, False otherwise

        Examples:
            >>> config = EmotionConfig()
            >>> config.is_valid_confidence(0.9)
            True
            >>> config.is_valid_confidence(-0.1)
            False
        """
        min_val, max_val = self.confidence_range
        return min_val <= value <= max_val

    def validate_confidence(self, value: float) -> EmotionValidationResult:
        """
        Validate a confidence value with detailed contextual information.

        Provides a rich validation result containing not just a boolean indicator,
        but also descriptive messages and context about the validation.

        Args:
            value: The confidence value to validate

        Returns:
            EmotionValidationResult: Structured validation result with context

        Examples:
            >>> config = EmotionConfig()
            >>> result = config.validate_confidence(0.9)
            >>> result.is_valid
            True
            >>> result.message
            'Confidence value 0.9 is within valid range (0.0, 1.0)'

            >>> result = config.validate_confidence(-0.1)
            >>> result.is_valid
            False
            >>> result.message
            'Confidence value -0.1 is outside valid range (0.0, 1.0)'
        """
        min_val, max_val = self.confidence_range
        is_valid = min_val <= value <= max_val

        if is_valid:
            message = (
                f"Confidence value {value} is within valid range ({min_val}, {max_val})"
            )
        else:
            message = f"Confidence value {value} is outside valid range ({min_val}, {max_val})"

        return EmotionValidationResult(
            is_valid=is_valid, message=message, value=value, range=self.confidence_range
        )

    def is_valid_emotion_category(self, category: str) -> bool:
        """
        Check if an emotion category is valid according to configuration.

        Validates that the provided category exists in the configured emotion keywords.

        Args:
            category: The emotion category to validate

        Returns:
            bool: True if the category exists in configuration, False otherwise

        Examples:
            >>> config = EmotionConfig()
            >>> config.is_valid_emotion_category("happiness")
            True
            >>> config.is_valid_emotion_category("confusion")  # Not in config
            False
        """
        return category in self.emotion_keywords

    def get_keywords_for_emotion(self, emotion: EmotionCategory) -> List[str]:
        """
        Get the list of keywords associated with an emotion category.

        Args:
            emotion: The emotion category to retrieve keywords for

        Returns:
            List of keywords associated with the emotion

        Raises:
            KeyError: If the emotion category doesn't exist in configuration

        Examples:
            >>> config = EmotionConfig()
            >>> config.get_keywords_for_emotion("happiness")
            ['happy', 'joy', 'delight', 'pleased', 'glad', 'excited']
        """
        return self.emotion_keywords[emotion]

    # ==========================================
    # SQL Template Properties
    # ==========================================
    # Maintained for backward compatibility

    @property
    def SQL_CREATE_WORD_EMOTION_TABLE(self) -> str:
        """
        SQL template for word emotion table creation.

        Returns:
            str: The SQL query for creating the word_emotion table
        """
        return self.sql_templates["create_word_emotion_table"]

    @property
    def SQL_CREATE_MESSAGE_EMOTION_TABLE(self) -> str:
        """
        SQL template for message emotion table creation.

        Returns:
            str: The SQL query for creating the message_emotion table
        """
        return self.sql_templates["create_message_emotion_table"]

    @property
    def SQL_INSERT_WORD_EMOTION(self) -> str:
        """
        SQL template for inserting word emotion data.

        Returns:
            str: The SQL query for inserting data into the word_emotion table
        """
        return self.sql_templates["insert_word_emotion"]

    @property
    def SQL_GET_WORD_EMOTION(self) -> str:
        """
        SQL template for retrieving word emotion data.

        Returns:
            str: The SQL query for retrieving data from the word_emotion table
        """
        return self.sql_templates["get_word_emotion"]

    @property
    def SQL_INSERT_MESSAGE_EMOTION(self) -> str:
        """
        SQL template for inserting message emotion data.

        Returns:
            str: The SQL query for inserting data into the message_emotion table
        """
        return self.sql_templates["insert_message_emotion"]

    @property
    def SQL_GET_MESSAGE_EMOTION(self) -> str:
        """
        SQL template for retrieving message emotion data.

        Returns:
            str: The SQL query for retrieving data from the message_emotion table
        """
        return self.sql_templates["get_message_emotion"]


__all__ = [
    # Main class
    "EmotionConfig",
    # Type definitions
    "EmotionValidationResult",
    "EmotionCategory",
    "EmotionKeywordsDict",
    # SQL Constants (for backward compatibility)
    "SQL_CREATE_WORD_EMOTION_TABLE",
    "SQL_CREATE_MESSAGE_EMOTION_TABLE",
    "SQL_INSERT_WORD_EMOTION",
    "SQL_GET_WORD_EMOTION",
    "SQL_INSERT_MESSAGE_EMOTION",
    "SQL_GET_MESSAGE_EMOTION",
]
