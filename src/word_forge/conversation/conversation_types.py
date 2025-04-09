"""
Type definitions for conversation management structures.
"""

from typing import List, Optional, TypedDict

from word_forge.emotion.emotion_types import EmotionAnalysisDict


class MessageDict(TypedDict):
    """Type definition for message data structure."""

    id: int
    speaker: str
    text: str
    timestamp: float
    emotion: Optional[EmotionAnalysisDict]


class ConversationDict(TypedDict):
    """Type definition for conversation data structure."""

    id: int
    status: str
    created_at: float
    updated_at: float
    messages: List[MessageDict]


__all__ = ["MessageDict", "ConversationDict"]
