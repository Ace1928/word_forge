import datetime
import random
import sqlite3
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

from word_forge.database.db_manager import DBManager
from word_forge.emotion.emotion_config import EmotionConfig
from word_forge.emotion.emotion_types import (
    EmotionalContext,
    EmotionAnalysisDict,
    FullEmotionAnalysisDict,
    MessageEmotionDict,
    WordEmotionDict,
)

# Optional LLM integration if available
from word_forge.parser.language_model import ModelState as LLMInterface

VADER_AVAILABLE = True
LLM_AVAILABLE = True


class EmotionError(Exception):
    """Exception raised for errors in the emotion analysis subsystem.

    Used for all emotion-related errors to provide a consistent interface
    with contextual information about what went wrong during emotional
    processing operations.

    Attributes:
        message: Explanation of the error
        timestamp: When the error occurred
        context: Additional contextual information about the error
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize an emotion error with detailed context.

        Args:
            message: Descriptive error message
            context: Optional dictionary of contextual information
        """
        self.timestamp = datetime.datetime.now().isoformat()
        self.context = context or {}
        super().__init__(message)


class EmotionManager:
    """Tracks emotional associations with words and conversation messages.

    This class manages the storage and retrieval of emotional data:
    - Word emotions: valence (positive/negative) and arousal levels
    - Message emotions: categorical emotion labels with confidence scores

    All emotional data is persistent and stored in SQLite tables.

    The emotion analysis uses a hybrid approach combining TextBlob, VADER
    (when available), and optional LLM integration for more accurate and
    nuanced sentiment detection in varied texts.

    Advanced recursive emotion processing enables deeper analysis of
    emotional relationships, meta-emotions, and emotional patterns.

    Attributes:
        db_manager: Database manager providing connection to storage
        config: Configuration for emotion analysis parameters
        VALENCE_RANGE: Valid range for valence values (-1.0 to 1.0 by default)
        AROUSAL_RANGE: Valid range for arousal values (0.0 to 1.0 by default)
        CONFIDENCE_RANGE: Valid range for confidence values (0.0 to 1.0 by default)
        EMOTION_KEYWORDS: Dictionary mapping emotion categories to keywords
        vader_weight: Weight given to VADER in hybrid analysis
        textblob_weight: Weight given to TextBlob in hybrid analysis
        llm_weight: Weight given to LLM analysis when available
        vader: VADER sentiment analyzer instance (if available)
        llm_interface: LLM interface for enhanced emotional analysis (if available)
        _recursive_processor: Cached instance of recursive emotion processor
    """

    def __init__(self, db_manager: DBManager) -> None:
        """Initialize the emotion tracking system.

        Establishes database connection, creates necessary tables, and
        initializes sentiment analysis tools. Sets up all configuration
        parameters to ensure consistent emotional measurement across
        the system.

        Args:
            db_manager: Database manager providing connection to storage

        Raises:
            EmotionError: If there's an issue initializing the emotion tables
                or connecting to required resources
        """
        self.db_manager = db_manager

        # Create config instance
        self.config = EmotionConfig()
        self._create_tables()

        # Initialize sentiment analyzers and LLM if available
        self._init_analysis_tools()

        # Access emotion constraints from centralized config instance
        self.VALENCE_RANGE = self.config.valence_range
        self.AROUSAL_RANGE = self.config.arousal_range
        self.CONFIDENCE_RANGE = self.config.confidence_range
        self.EMOTION_KEYWORDS = self.config.emotion_keywords

        # Hybrid analysis weights
        self.vader_weight = self.config.vader_weight
        self.textblob_weight = self.config.textblob_weight
        self.llm_weight = getattr(self.config, "llm_weight", 0.0)

        # Initialize recursive emotion processor lazily
        self._recursive_processor = None

    @property
    def recursive_processor(self):
        """Lazy initialization of recursive emotion processor.

        Returns:
            RecursiveEmotionProcessor: Emotion processor for recursive analysis
        """
        if self._recursive_processor is None:
            # Import here to avoid circular import issues
            from word_forge.emotion.emotion_processor import RecursiveEmotionProcessor

            self._recursive_processor = RecursiveEmotionProcessor(
                db_manager=self.db_manager, emotion_manager=self
            )
        return self._recursive_processor

    def _init_analysis_tools(self) -> None:
        """Initialize all available sentiment and emotion analysis tools.

        Sets up:
        1. VADER sentiment analyzer if available - specialized for social media
        2. LLM interface if available - for deeper semantic understanding

        Each tool is activated according to configuration settings to allow
        flexible deployment in various environments.
        """
        # Initialize VADER if available and enabled
        self.vader = (
            SentimentIntensityAnalyzer()
            if VADER_AVAILABLE and self.config.enable_vader
            else None
        )

        # Initialize LLM interface if available and enabled
        self.llm_interface = None
        if LLM_AVAILABLE and getattr(self.config, "enable_llm", False):
            try:
                self.llm_interface = LLMInterface()
            except Exception as e:
                print(
                    f"Warning: LLM initialization failed (continuing without LLM): {e}"
                )

    def init_analysis_tools(self) -> None:
        """Reinitialize analysis tools.

        This method allows for reinitialization of the sentiment analysis
        tools, useful in cases where the configuration has changed or
        the system needs to refresh its resources.
        """
        self._init_analysis_tools()

    @contextmanager
    def _db_connection(self):
        """Create a database connection using the DBManager's path.

        Creates and manages a database connection with SQLite's Row factory
        for dictionary-like access to results. Automatically closes the
        connection when exiting the context.

        Yields:
            sqlite3.Connection: An active database connection

        Example:
            ```python
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM word_emotion")
                results = cursor.fetchall()
            # Connection automatically closed after block
            ```
        """
        conn = sqlite3.connect(self.db_manager.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _create_tables(self) -> None:
        """Initialize database tables for emotion storage.

        Creates two tables if they don't exist:
        - word_emotion: Links words to valence and arousal scores
        - message_emotion: Links messages to categorical emotion labels

        Table schemas are defined in the EmotionConfig class to ensure
        consistency across system components.

        Raises:
            EmotionError: If there's an issue creating the emotion tables
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(self.config.SQL_CREATE_WORD_EMOTION_TABLE)
                cursor.execute(self.config.SQL_CREATE_MESSAGE_EMOTION_TABLE)
                conn.commit()
        except sqlite3.Error as e:
            raise EmotionError(
                f"Failed to initialize emotion tables: {e}",
                {"operation": "create_tables", "db_path": self.db_manager.db_path},
            ) from e

    @lru_cache(maxsize=256)
    def set_word_emotion(self, word_id: int, valence: float, arousal: float) -> None:
        """Store or update emotional values for a word.

        Assigns valence (positivity/negativity) and arousal (intensity)
        values to a word, clamping values to valid ranges defined in
        configuration.

        Args:
            word_id: Database ID of the target word
            valence: Emotional valence (-1.0 to 1.0)
            arousal: Emotional arousal/intensity (0.0 to 1.0)

        Raises:
            EmotionError: If the database operation fails

        Example:
            ```python
            # Set "happy" as strongly positive with moderate arousal
            word_id = db_manager.get_word_id("happy")
            emotion_manager.set_word_emotion(word_id, 0.8, 0.6)
            ```
        """
        valence = max(self.VALENCE_RANGE[0], min(self.VALENCE_RANGE[1], valence))
        arousal = max(self.AROUSAL_RANGE[0], min(self.AROUSAL_RANGE[1], arousal))

        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    self.config.SQL_INSERT_WORD_EMOTION,
                    (word_id, valence, arousal, time.time()),
                )
                conn.commit()
        except sqlite3.Error as e:
            raise EmotionError(
                f"Failed to store word emotion: {e}",
                {"word_id": word_id, "valence": valence, "arousal": arousal},
            ) from e

    @lru_cache(maxsize=256)
    def get_word_emotion(self, word_id: int) -> Optional[WordEmotionDict]:
        """Retrieve emotional data for a word.

        Args:
            word_id: Database ID of the target word

        Returns:
            Dictionary containing emotional data (valence, arousal, timestamp),
            or None if no data exists for the word

        Raises:
            EmotionError: If the database operation fails

        Example:
            ```python
            word_id = db_manager.get_word_id("happy")
            emotion_data = emotion_manager.get_word_emotion(word_id)
            if emotion_data:
                print(f"Valence: {emotion_data['valence']}")
                print(f"Arousal: {emotion_data['arousal']}")
            ```
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(self.config.SQL_GET_WORD_EMOTION, (word_id,))
                row = cursor.fetchone()

                if row:
                    return cast(WordEmotionDict, dict(row))
                return None
        except sqlite3.Error as e:
            raise EmotionError(
                f"Failed to retrieve word emotion: {e}", {"word_id": word_id}
            ) from e

    @lru_cache(maxsize=256)
    def set_message_emotion(
        self, message_id: int, label: str, confidence: float = 1.0
    ) -> None:
        """Tag a message with an emotion label.

        Records the emotional classification of a message with a confidence
        score, ensuring the confidence value is within valid range.

        Args:
            message_id: Database ID of the target message
            label: Emotional category name (e.g., 'happy', 'sad')
            confidence: Certainty level of the emotion (0.0 to 1.0)

        Raises:
            EmotionError: If the database operation fails

        Example:
            ```python
            # Tag message as expressing anger with high confidence
            emotion_manager.set_message_emotion(1042, "anger", 0.85)
            ```
        """
        confidence = max(
            self.CONFIDENCE_RANGE[0], min(self.CONFIDENCE_RANGE[1], confidence)
        )

        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    self.config.SQL_INSERT_MESSAGE_EMOTION,
                    (message_id, label, confidence, time.time()),
                )
                conn.commit()
        except sqlite3.Error as e:
            raise EmotionError(
                f"Failed to store message emotion: {e}",
                {"message_id": message_id, "label": label, "confidence": confidence},
            ) from e

    @lru_cache(maxsize=256)
    def get_message_emotion(self, message_id: int) -> Optional[MessageEmotionDict]:
        """Retrieve emotional data for a message.

        Args:
            message_id: Database ID of the target message

        Returns:
            Dictionary containing emotion label, confidence and timestamp,
            or None if no data exists for the message

        Raises:
            EmotionError: If the database operation fails

        Example:
            ```python
            emotion_data = emotion_manager.get_message_emotion(1042)
            if emotion_data:
                print(f"Emotion: {emotion_data['label']}")
                print(f"Confidence: {emotion_data['confidence']}")
            ```
        """
        try:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(self.config.SQL_GET_MESSAGE_EMOTION, (message_id,))
                row = cursor.fetchone()

                if row:
                    return cast(MessageEmotionDict, dict(row))
                return None
        except sqlite3.Error as e:
            raise EmotionError(
                f"Failed to retrieve message emotion: {e}", {"message_id": message_id}
            ) from e

    @lru_cache(maxsize=256)
    def _analyze_with_vader(self, text: str) -> Tuple[float, float]:
        """Analyze text sentiment using VADER.

        VADER is specifically tuned for social media content and handles
        emoticons, slang, and capitalization particularly well.

        Args:
            text: Text to analyze for emotional content

        Returns:
            Tuple of (valence, arousal) values from VADER

        Note:
            Returns (0.0, 0.0) if VADER is not available
        """
        if not self.vader:
            return 0.0, 0.0

        # Get sentiment scores from VADER analyzer
        vader_scores = self.vader.polarity_scores(text)

        # Map VADER compound score (-1 to 1) directly to valence
        valence = vader_scores["compound"]

        # Estimate arousal from VADER's intensity measures
        # Higher intensity (positive + negative) suggests higher arousal
        intensity = vader_scores["pos"] + vader_scores["neg"]
        arousal = min(intensity * 1.5, 1.0)  # Scale and cap to our range

        return valence, arousal

    @lru_cache(maxsize=256)
    def _analyze_with_llm(self, text: str) -> Tuple[float, float, Dict[str, float]]:
        """Analyze text sentiment using LLM for deeper emotional understanding.

        Uses structured prompting of the language model to extract nuanced
        emotional dimensions beyond basic sentiment, including emotional
        intensity, specific emotional categories, and emotional undertones.

        Args:
            text: Text to analyze for emotional content

        Returns:
            Tuple of (valence, arousal, emotion_dimensions) where emotion_dimensions
            is a dictionary of additional emotional attributes with their strengths

        Note:
            Returns (0.0, 0.0, {}) if LLM integration is not available
        """
        if not self.llm_interface:
            return 0.0, 0.0, {}

        try:
            # Structured query to the LLM for emotion analysis
            prompt = f"""Analyze the emotional content of this text:
"{text}"

Return the analysis as a JSON object with these fields:
- valence: a float from -1.0 (negative) to 1.0 (positive)
- arousal: a float from 0.0 (calm) to 1.0 (excited)
- primary_emotion: the dominant emotion as a string
- emotion_dimensions: an object with emotional attributes and their strengths (0.0-1.0)
- emotional_undertones: an array of subtle emotional qualities present
"""
            # Get response from LLM and ensure it's a dictionary
            response_raw = self.llm_interface.query(prompt)

            # Process response into dict format - handle both string and dict responses
            response: Dict[str, Any] = {}
            if isinstance(response_raw, dict):
                response = cast(Dict[str, Any], response_raw)
            elif isinstance(response_raw, str):
                import json

                try:
                    response = cast(Dict[str, Any], json.loads(response_raw))
                except json.JSONDecodeError:
                    # If not valid JSON, create minimal response
                    response = {}

            # Extract the core emotional dimensions with safe type handling
            valence_raw: Any = response.get("valence", 0.0)
            try:
                valence = float(valence_raw)
            except (ValueError, TypeError):
                valence = 0.0

            arousal_raw: Any = response.get("arousal", 0.0)
            try:
                arousal = float(arousal_raw)
            except (ValueError, TypeError):
                arousal = 0.0

            # Normalized to expected ranges
            valence = max(self.VALENCE_RANGE[0], min(self.VALENCE_RANGE[1], valence))
            arousal = max(self.AROUSAL_RANGE[0], min(self.AROUSAL_RANGE[1], arousal))

            # Additional emotional dimensions
            emotion_dimensions_raw: Any = response.get("emotion_dimensions", {})
            emotion_dimensions: Dict[str, float] = {}

            # Ensure emotion_dimensions is a valid dictionary with float values
            if isinstance(emotion_dimensions_raw, dict):
                # Cast to Dict[Any, Any] to help type checker understand dictionary structure
                typed_dimensions = cast(Dict[Any, Any], emotion_dimensions_raw)
                for key_raw, value_raw in typed_dimensions.items():
                    key = str(key_raw)
                    try:
                        emotion_dimensions[key] = float(value_raw)
                    except (ValueError, TypeError):
                        emotion_dimensions[key] = 0.0

            return valence, arousal, emotion_dimensions

        except Exception as e:
            # Log error but don't fail - continue with other analyzers
            print(f"LLM emotion analysis failed (continuing with other methods): {e}")
            return 0.0, 0.0, {}

    @lru_cache(maxsize=256)
    def analyze_text_emotion(self, text: str) -> Tuple[float, float]:
        """Analyze text to determine emotional valence and arousal.

        Uses a hybrid approach combining TextBlob, VADER (when available),
        and LLM (when available) for comprehensive sentiment analysis across
        different text styles. Results are weighted and combined according to
        configuration parameters.

        Args:
            text: Text to analyze for emotional content

        Returns:
            Tuple of (valence, arousal) values normalized to proper ranges

        Example:
            ```python
            valence, arousal = emotion_manager.analyze_text_emotion(
                "I'm feeling really excited about this new project!"
            )
            print(f"Valence: {valence:.2f}, Arousal: {arousal:.2f}")
            # Might output: Valence: 0.82, Arousal: 0.75
        """

        # TextBlob analysis (always available)
        blob = TextBlob(text)
        # Extract sentiment properties safely
        try:
            # Access properties directly from returned sentiment namedtuple
            sentiment = blob.sentiment
            textblob_valence = float(getattr(sentiment, "polarity", 0.0))
            subjectivity = float(getattr(sentiment, "subjectivity", 0.0))
        except (AttributeError, TypeError, ValueError):
            textblob_valence = 0.0
            subjectivity = 0.0

        exclamation_count = text.count("!")
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        # Initialize weighted results
        valence_components: List[Tuple[float, float]] = [
            (textblob_valence, self.textblob_weight)
        ]
        arousal_factors: List[Tuple[float, float]] = [
            (subjectivity, 1.0),
            (min(exclamation_count / 5, 1.0), 1.0),
            (uppercase_ratio * 2, 1.0),
        ]

        # VADER analysis (when available)
        if self.vader:
            vader_valence, vader_arousal = self._analyze_with_vader(text)
            valence_components.append((vader_valence, self.vader_weight))
            arousal_factors.append(
                (vader_arousal * 2, 2.0)
            )  # VADER's arousal weighted higher

        # LLM analysis (when available)
        if self.llm_interface and self.llm_weight > 0:
            llm_valence, llm_arousal, _ = self._analyze_with_llm(text)
            valence_components.append((llm_valence, self.llm_weight))
            arousal_factors.append((llm_arousal, self.llm_weight * 1.5))

        # Weighted fusion of all analyzers
        total_valence_weight = sum(weight for _, weight in valence_components)
        if total_valence_weight > 0:
            valence = (
                sum(val * weight for val, weight in valence_components)
                / total_valence_weight
            )
        else:
            valence = textblob_valence  # Fallback

        # Combine all arousal factors with their respective weights
        total_arousal_weight = sum(weight for _, weight in arousal_factors)
        if total_arousal_weight > 0:
            arousal = (
                sum(val * weight for val, weight in arousal_factors)
                / total_arousal_weight
            )
        else:
            # Original arousal calculation as fallback
            arousal = sum(val for val, _ in arousal_factors) / len(arousal_factors)

        # Ensure values are within bounds
        arousal = min(arousal, 1.0)
        valence = max(min(valence, 1.0), -1.0)

        return valence, arousal

    def _count_keyword_occurrences(self, text: str) -> Dict[str, int]:
        """Count emotion keyword occurrences in a text.

        Searches for emotion-specific keywords defined in the configuration,
        using case-insensitive matching to maximize detection.

        Args:
            text: Text to analyze for emotion keywords

        Returns:
            Dictionary mapping emotion categories to keyword occurrence counts

        Example:
            ```python
            counts = emotion_manager._count_keyword_occurrences(
                "I'm feeling happy and excited today!"
            )
            # Might return: {'happiness': 2, 'anger': 0, 'sadness': 0, ...}
            ```
        """
        text_lower = text.lower()
        result: Dict[str, int] = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            if isinstance(keywords, list):
                # Cast keywords to List[str] for type checking
                str_keywords: List[str] = cast(List[str], keywords)
                result[emotion] = sum(text_lower.count(word) for word in str_keywords)
            else:
                result[emotion] = 0
        return result

    def classify_emotion(self, text: str) -> Tuple[str, float]:
        """Classify text into basic emotion categories.

        Uses a hybrid approach combining sentiment analysis, keyword detection,
        and optional LLM classification to determine the most likely emotion
        expressed in the text, along with a confidence score.

        Args:
            text: Text to analyze for emotional content

        Returns:
            Tuple of (emotion_label, confidence)

        Example:
            ```python
            emotion, confidence = emotion_manager.classify_emotion(
                "I'm absolutely thrilled about the news!"
            )
            print(f"Emotion: {emotion}, Confidence: {confidence:.2f}")
            # Might output: Emotion: happiness, Confidence: 0.85
            ```
        """
        # Try LLM classification if available
        if self.llm_interface and self.llm_weight > 0:
            try:
                _, _, emotion_dims = self._analyze_with_llm(text)
                if emotion_dims and "primary_emotion" in emotion_dims:
                    primary = str(emotion_dims["primary_emotion"])
                    confidence = emotion_dims.get("confidence", 0.7)
                    # Map to our standard emotion categories if needed
                    if primary in self.EMOTION_KEYWORDS:
                        return primary, min(confidence, 1.0)
            except Exception:
                # Continue with regular classification if LLM fails
                pass

        # Get valence for sentiment direction
        valence, _ = self.analyze_text_emotion(text)

        # Count keyword occurrences for each emotion
        keyword_counts = self._count_keyword_occurrences(text)

        # Combine keyword counts with valence for better classification
        if abs(valence) < 0.2:
            # Near-neutral sentiment - rely more on keywords or default to neutral
            emotion = max(keyword_counts.items(), key=lambda x: x[1])[0]
            if keyword_counts[emotion] == 0:
                emotion = "neutral"
        else:
            # Strong sentiment - choose between positive or negative emotions
            if valence > 0:
                positive_emotions = ["happiness", "surprise"]
                candidates = {e: keyword_counts[e] for e in positive_emotions}
                emotion = max(candidates.items(), key=lambda x: x[1])[0]
                if candidates[emotion] == 0:
                    emotion = "happiness"  # Default positive
            else:
                negative_emotions = ["sadness", "anger", "fear", "disgust"]
                candidates = {e: keyword_counts[e] for e in negative_emotions}
                emotion = max(candidates.items(), key=lambda x: x[1])[0]
                if candidates[emotion] == 0:
                    emotion = "sadness"  # Default negative

        # Calculate confidence based on keyword strength and valence agreement
        keyword_strength = (
            min(keyword_counts[emotion] / 3, 1.0)
            if keyword_counts[emotion] > 0
            else self.config.min_keyword_confidence
        )
        confidence = 0.4 + (self.config.keyword_match_weight * keyword_strength)

        # Increase confidence if valence strongly agrees with emotion polarity
        emotion_is_positive = emotion in ["happiness", "surprise"]
        valence_matches_emotion = (valence > 0.3 and emotion_is_positive) or (
            valence < -0.3 and not emotion_is_positive
        )
        if valence_matches_emotion:
            confidence = min(confidence + 0.15, 1.0)

        return emotion, confidence

    def process_message(self, message_id: int, text: str) -> EmotionAnalysisDict:
        """Process a message to identify and store its emotional content.

        Performs emotion classification on the message text and stores
        the results in the database for later retrieval and analysis.

        Args:
            message_id: Database ID of the message
            text: Message text to analyze

        Returns:
            Dictionary containing the identified emotion data

        Raises:
            EmotionError: If storing the emotion data fails

        Example:
            ```python
            result = emotion_manager.process_message(
                1042, "I'm really disappointed with the service."
            )
            print(f"Detected emotion: {result['emotion_label']}")
            print(f"Confidence: {result['confidence']}")
            ```
        """
        # Classify the emotion in the text
        emotion_label, confidence = self.classify_emotion(text)

        # Store the emotion data
        self.set_message_emotion(message_id, emotion_label, confidence)

        return {"emotion_label": emotion_label, "confidence": confidence}

    def analyze_term_recursively(
        self, term: str, context: Optional[Union[str, EmotionalContext]] = None
    ) -> FullEmotionAnalysisDict:
        """Perform deep emotional analysis on a term using recursive processing.

        This method provides a comprehensive emotional analysis beyond basic
        sentiment analysis, including meta-emotions, emotional patterns, and
        contextual emotional responses. Uses a recursive model that explores
        emotional associations at multiple levels.

        Args:
            term: The word or phrase to analyze
            context: Optional emotional context or context name

        Returns:
            Dictionary containing comprehensive emotional analysis including:
            - Primary emotion classification
            - Dimensional emotional measurements
            - Meta-emotions (emotions about emotions)
            - Emotional patterns and sequences
            - Recursive analysis depth

        Note:
            This is a more computationally intensive analysis than basic
            sentiment classification and should be used selectively.

        Example:
            ```python
            deep_analysis = emotion_manager.analyze_term_recursively("betrayal")
            print(f"Primary emotion: {deep_analysis['emotion_label']}")
            print(f"Dimensions: {deep_analysis['dimensions']}")
            print(f"Meta-emotions: {[m['label'] for m in deep_analysis['meta_emotions']]}")
            ```
        """
        # Prepare context if provided as a string
        actual_context = None
        if isinstance(context, str):
            actual_context = self.recursive_processor.get_context(context)
            if not actual_context and context:
                # Try to create a domain-specific context
                actual_context = self.recursive_processor.create_context_for_domain(
                    context
                )
        elif isinstance(context, EmotionalContext):
            actual_context = context

        # Process the term using recursive processor
        concept = self.recursive_processor.process_term(term, actual_context)

        # Get the basic emotion category and confidence
        emotion_label, confidence = self.classify_emotion(term)

        # Enhance with LLM-derived insights if available
        additional_insights: Dict[str, Any] = {}
        if self.llm_interface and self.llm_weight > 0:
            try:
                prompt = f"""Analyze the emotional associations of the term '{term}' in depth.

Focus especially on:
1. Meta-emotions (emotions about emotions)
2. Emotional patterns or sequences this term might evoke
3. Cultural and contextual emotional variations
4. Temporal dynamics (how emotions related to this term might change over time)

Format your response as a structured JSON with insights in each category."""

                insights = self.llm_interface.query(prompt)
                if isinstance(insights, (dict, str)) and insights:
                    additional_insights = insights if isinstance(insights, dict) else {}
            except Exception as e:
                # Continue without LLM insights if there's an error
                print(f"LLM insights generation failed: {e}")

        # Create the return structure
        result: Dict[str, Any] = {
            "emotion_label": emotion_label,
            "confidence": confidence,
            "concept": concept.as_dict(),
            "dimensions": concept.primary_emotion.as_dict(),
            "meta_emotions": [
                {"label": label, "dimensions": emotion.as_dict()}
                for label, emotion in concept.meta_emotions
            ],
            "patterns": {
                pattern_type: [emotion.as_dict() for emotion in sequence]
                for pattern_type, sequence in concept.emotional_patterns.items()
            },
            "recursive_depth": concept.recursive_depth,
        }

        # Add LLM insights if available
        if additional_insights:
            result["enhanced_insights"] = additional_insights

        return cast(FullEmotionAnalysisDict, result)

    def analyze_emotional_relationship(
        self, term1: str, term2: str, relationship_type: str
    ) -> float:
        """Analyze the emotional relationship between two terms.

        Evaluates the strength and nature of emotional relationships between
        concepts, allowing for sophisticated emotional reasoning.

        Args:
            term1: First term to analyze
            term2: Second term to analyze
            relationship_type: Type of relationship to assess

        Returns:
            Strength of the relationship (0.0-1.0)

        Relationship types include:
        - emotional_synonym: Similar emotional quality
        - emotional_antonym: Opposite emotional quality
        - intensifies/diminishes: One term modifies intensity
        - emotional_component/composite: Part-whole relationships
        - evokes: One term triggers the emotion of another
        - meta_emotion: Emotion about an emotion

        Example:
            ```python
            # Check if "ecstatic" intensifies "happy"
            strength = emotion_manager.analyze_emotional_relationship(
                "ecstatic", "happy", "intensifies"
            )
            print(f"Intensification strength: {strength:.2f}")
            # Might output: Intensification strength: 0.87
            ```
        """
        # Use enhanced LLM analysis if available for more nuanced relationship assessment
        if self.llm_interface and self.llm_weight > 0:
            try:
                prompt = f"""Analyze the emotional relationship between '{term1}' and '{term2}'
specifically focusing on the relationship type: '{relationship_type}'.

Assess the strength of this relationship on a scale from 0.0 (no relationship) to 1.0 (strongest possible relationship).

Consider:
- Semantic similarity and difference
- Emotional intensity differences
- Cultural and contextual factors
- Linguistic patterns of usage

Return only a numeric value between 0.0 and 1.0 representing the strength."""

                result = self.llm_interface.query(prompt)

                # Get base strength from recursive processor for fallback or blending
                base_strength = self.recursive_processor.analyze_relationship(
                    term1, term2, relationship_type
                )

                # Extract numeric value from result
                if result is not None:
                    try:
                        strength = float(result.strip())
                        # Ensure proper range
                        strength = max(0.0, min(1.0, strength))

                        # Weighted average
                        return strength * self.llm_weight + base_strength * (
                            1 - self.llm_weight
                        )
                    except (ValueError, TypeError):
                        # Fall back to recursive processor if parsing fails
                        pass
            except Exception:
                # Fall back to recursive processor if LLM fails
                pass

        # Default recursive processor analysis
        return self.recursive_processor.analyze_relationship(
            term1, term2, relationship_type
        )

    def create_emotional_context(
        self,
        domain: Optional[str] = None,
        cultural_factors: Optional[Dict[str, float]] = None,
        situational_factors: Optional[Dict[str, float]] = None,
        temporal_factors: Optional[Dict[str, float]] = None,
        domain_specific: Optional[Dict[str, float]] = None,
    ) -> EmotionalContext:
        """Create a custom emotional context for analysis.

        Emotional contexts provide frameworks for interpreting emotions
        within specific domains, cultures, or situations. They adjust the
        baseline emotional responses to account for contextual factors.

        Args:
            domain: Optional domain name for predefined contexts
            cultural_factors: Optional cultural dimension adjustments
            situational_factors: Optional situational adjustments
            temporal_factors: Optional temporal dimension adjustments
            domain_specific: Optional domain-specific adjustments

        Returns:
            An emotional context object for use in recursive analysis

        Example:
            ```python
            # Create a context for analyzing medical terminology
            medical_context = emotion_manager.create_emotional_context(
                domain="medical",
                situational_factors={"clinical_setting": 0.8, "emergency": 0.2}
            )

            # Analyze term within this context
            analysis = emotion_manager.analyze_term_recursively(
                "diagnosis", medical_context
            )
            ```
        """
        # If domain is provided, start with domain-specific context
        if domain:
            context = self.recursive_processor.create_context_for_domain(domain)
        else:
            context = EmotionalContext()

        # Apply any provided factor dictionaries
        if cultural_factors:
            context.cultural_factors.update(cultural_factors)
        if situational_factors:
            context.situational_factors.update(situational_factors)
        if temporal_factors:
            context.temporal_factors.update(temporal_factors)
        if domain_specific:
            context.domain_specific.update(domain_specific)

        return context

    def register_emotional_context(self, name: str, context: EmotionalContext) -> None:
        """Register a named emotional context for reuse.

        Stores a context in the system's registry for later retrieval,
        allowing consistent application of the same contextual factors
        across multiple analyses.

        Args:
            name: Name to identify this context
            context: The emotional context to register

        Example:
            ```python
            # Create and register a "clinical" context
            clinical = emotion_manager.create_emotional_context(
                domain="medical",
                situational_factors={"professional": 0.9, "emergency": 0.1}
            )
            emotion_manager.register_emotional_context("clinical", clinical)

            # Later retrieve by name
            analysis = emotion_manager.analyze_term_recursively(
                "prognosis", "clinical"  # Using string name to reference context
            )
            ```
        """
        self.recursive_processor.register_context(name, context)

    def get_emotional_context(self, name: str) -> Optional[EmotionalContext]:
        """Retrieve a registered emotional context by name.

        Args:
            name: Name of the context to retrieve

        Returns:
            The emotional context if found, None otherwise

        Example:
            ```python
            # Retrieve a previously registered context
            clinical_context = emotion_manager.get_emotional_context("clinical")
            if clinical_context:
                # Use the context
                emotion_manager.analyze_term_recursively("diagnosis", clinical_context)
            ```
        """
        return self.recursive_processor.get_context(name)

    def enrich_word_emotions(self, term: str, word_id: int) -> bool:
        """Enrich emotional data for a word using all available analysis methods.

        Combines basic sentiment analysis with deeper LLM-based insights when
        available to provide comprehensive emotional profiling of a term.

        Args:
            term: The word to analyze
            word_id: Database ID of the word

        Returns:
            Boolean indicating whether enrichment was successful

        Example:
            ```python
            word_id = db_manager.get_word_id("nostalgia")
            emotion_manager.enrich_word_emotions("nostalgia", word_id)
            # Retrieves and stores enriched emotional data for "nostalgia"
            ```
        """
        try:
            # Basic sentiment analysis
            valence, arousal = self.analyze_text_emotion(term)

            # If LLM is available, get deeper insights
            if self.llm_interface and self.llm_weight > 0:
                try:
                    prompt = f"""Analyze the emotional qualities of the word '{term}'.

Provide a detailed analysis of:
1. Valence (positivity/negativity) on a scale from -1.0 to 1.0
2. Arousal (intensity/energy) on a scale from 0.0 to 1.0
3. The complexity and nuance of emotions typically associated with this term
4. Common contexts where this word carries emotional weight
5. Cultural variations in emotional perception of this term

Focus on emotional qualities rather than definitions."""

                    insights = self.llm_interface.query(prompt)

                    # Extract more precise valence/arousal from LLM if possible
                    if insights is not None:
                        llm_valence, llm_arousal, _ = self._analyze_with_llm(
                            term + ". " + str(insights)
                        )

                        # Blend basic and LLM-enhanced values
                        if isinstance(llm_valence, float) and isinstance(
                            llm_arousal, float
                        ):
                            valence = (valence + llm_valence) / 2
                            arousal = (arousal + llm_arousal) / 2

                except Exception as e:
                    # Continue with basic analysis if LLM fails
                    print(f"LLM word enrichment failed for '{term}': {e}")

            # Store the enriched emotional data
            self.set_word_emotion(word_id, valence, arousal)

            # If we have recursive processing capability, do deeper analysis
            if self._recursive_processor:
                # Just trigger the processing - results are cached internally
                self.analyze_term_recursively(term)

            return True

        except Exception as e:
            print(f"Error enriching word emotions for '{term}': {e}")
            return False


def main() -> None:
    """Demonstrate EmotionManager functionality with comprehensive emotion analysis."""
    # Initialize with actual DBManager instead of temporary database
    db_manager = DBManager()

    # Create database tables if they don't exist
    db_manager.create_tables()

    # Initialize emotion manager with VADER and LLM if available
    emotion_mgr = EmotionManager(db_manager)

    # Ensure LLM and VADER are enabled if available
    if VADER_AVAILABLE:
        emotion_mgr.config.enable_vader = True
    if LLM_AVAILABLE:
        setattr(emotion_mgr.config, "enable_llm", True)
        setattr(emotion_mgr.config, "llm_weight", 0.6)  # Give significant weight to LLM

    # Re-initialize analysis tools with updated config
    emotion_mgr.init_analysis_tools()

    # Ensure some emotion-related words exist for demonstration
    basic_emotions = [
        (
            "happiness",
            "A state of well-being and contentment",
            "noun",
            ["Finding true happiness is a journey, not a destination."],
        ),
        (
            "sadness",
            "The condition of being sad; sorrow or grief",
            "noun",
            ["She couldn't hide her sadness after the loss."],
        ),
        (
            "anger",
            "A strong feeling of displeasure or hostility",
            "noun",
            ["He managed his anger through meditation techniques."],
        ),
        (
            "fear",
            "An unpleasant emotion caused by anticipation of danger",
            "noun",
            ["Fear of failure kept him from pursuing his dreams."],
        ),
        (
            "surprise",
            "An unexpected event or piece of information",
            "noun",
            ["The birthday party was a complete surprise."],
        ),
    ]

    # Insert basic emotion words if needed
    print("Ensuring emotion vocabulary exists in database...")
    for term, definition, pos, examples in basic_emotions:
        try:
            db_manager.insert_or_update_word(term, definition, pos, examples)
            print(f"  ✓ {term}")
        except Exception as e:
            print(f"  ✗ Could not add '{term}': {e}")

    # Process and track emotions for all words in the database
    try:
        # Get all words from the database
        all_words = db_manager.get_all_words()
        word_count = len(all_words)
        print(f"\nEnriching emotions for all {word_count} words in the database...")

        # First pass: Enrich each word with full emotional data
        enriched_count = 0
        for word_data in all_words:
            word_id = word_data["id"]
            term = word_data["term"]

            # Use the comprehensive enrichment method instead of basic analysis
            emotion_mgr.enrich_word_emotions(term, word_id)

            enriched_count += 1
            if enriched_count % 50 == 0:
                print(f"  Enriched {enriched_count}/{word_count} words...")

        print(f"Completed enriching all {word_count} words.")

        # Second pass: Ensure recursive analysis for all terms
        print(f"\nPerforming recursive analysis on all {word_count} words...")
        recursive_count = 0
        for word_data in all_words:
            term = word_data["term"]

            # Process term recursively to build emotional relationships
            emotion_mgr.analyze_term_recursively(term)

            recursive_count += 1
            if recursive_count % 50 == 0:
                print(f"  Recursively analyzed {recursive_count}/{word_count} words...")

        print(f"Completed recursive analysis for all {word_count} words.")

        # Display a random sample of entries
        sample_size = min(100, word_count)
        sample_indices = random.sample(range(word_count), sample_size)
        sample_words = [all_words[i] for i in sample_indices]

        print(
            f"\nDisplaying random sample of {sample_size} fully enriched word emotions:"
        )
        print("-" * 60)

        for i, word_data in enumerate(sample_words):
            word_id = word_data["id"]
            term = word_data["term"]
            definition = word_data["definition"] or ""

            # Extract usage examples
            usage_examples = []
            if word_data["usage_examples"]:
                usage_str = word_data["usage_examples"]
                usage_examples = usage_str.split("; ") if usage_str else []

            # Retrieve stored emotion data
            stored_emotion = emotion_mgr.get_word_emotion(word_id)
            if not stored_emotion:
                continue

            # Display results
            print(f"Word {i+1}/{sample_size}: '{term}'")
            print(
                f"  - Word Emotion: Valence={stored_emotion['valence']:.2f}, Arousal={stored_emotion['arousal']:.2f}"
            )

            if definition:
                def_valence, def_arousal = emotion_mgr.analyze_text_emotion(definition)
                def_emotion, def_confidence = emotion_mgr.classify_emotion(definition)
                print(f"  - Definition: '{definition}'")
                print(f"    Valence={def_valence:.2f}, Arousal={def_arousal:.2f}")
                print(
                    f"    Classified as: {def_emotion} (confidence: {def_confidence:.2f})"
                )

            # Analyze an example if available
            if usage_examples:
                example = usage_examples[0]
                ex_valence, ex_arousal = emotion_mgr.analyze_text_emotion(example)
                ex_emotion, ex_confidence = emotion_mgr.classify_emotion(example)
                print(f"  - Example: '{example}'")
                print(f"    Valence={ex_valence:.2f}, Arousal={ex_arousal:.2f}")
                print(
                    f"    Classified as: {ex_emotion} (confidence: {ex_confidence:.2f})"
                )

            # Show recursive emotion data if available
            try:
                recursive_data = emotion_mgr.analyze_term_recursively(term)
                print("  - Recursive Analysis:")
                print(
                    f"    Recursive depth: {recursive_data.get('recursive_depth', 'N/A')}"
                )
                meta_emotions = recursive_data.get("meta_emotions", [])
                if meta_emotions and len(meta_emotions) > 0:
                    print(
                        f"    Meta-emotions: {meta_emotions[0].get('label', 'unknown')}, ..."
                    )
                patterns = recursive_data.get("patterns", {})
                if patterns and len(patterns) > 0:
                    pattern_keys = list(patterns.keys())
                    print(
                        f"    Patterns: {pattern_keys[0] if pattern_keys else 'none'}, ..."
                    )
            except Exception:
                pass

            print("-" * 60)

    except Exception as e:
        print(f"Error processing word emotions: {e}")

    # Display analyzer availability
    llm_status = "✓ LLM" if LLM_AVAILABLE else "✗ LLM (not available)"
    vader_status = "✓ VADER" if VADER_AVAILABLE else "✗ VADER (not available)"
    print(f"\nEmotion Analyzers: TextBlob + {vader_status} + {llm_status}")

    # Example: Analyze and track emotions for messages
    messages = [
        (100, "I'm so happy today! Everything is going GREAT!"),
        (101, "I feel sad and disappointed about the results."),
        (102, "This product is amazing and exceeds expectations."),
        (103, "I'm furious about the terrible customer service."),
        (104, "Just received the package, it looks fine."),
        (105, "OMG this is AWESOME!!! 😊 I love it so much!"),
    ]

    print("\nMessage Emotion Analysis:")
    print("-" * 60)
    for message_id, text in messages:
        # Analyze raw emotional dimensions
        valence, arousal = emotion_mgr.analyze_text_emotion(text)

        # Classify into emotion category
        emotion_data = emotion_mgr.process_message(message_id, text)

        # Retrieve stored data to verify
        stored_emotion = emotion_mgr.get_message_emotion(message_id)

        print(f'Message: "{text}"')
        print(f"Valence: {valence:.2f}, Arousal: {arousal:.2f}")
        print(
            f"Classified as: {emotion_data['emotion_label']} (confidence: {emotion_data['confidence']:.2f})"
        )
        print(f"Stored data: {stored_emotion}")
        print("-" * 60)

    # Demonstrate emotional relationships
    print("\nEmotional Relationship Analysis:")
    print("-" * 60)

    relationship_tests = [
        ("happiness", "joy", "emotional_synonym"),
        ("happiness", "sadness", "emotional_antonym"),
        ("excited", "ecstatic", "intensifies"),
        ("anxiety", "worry", "emotional_component"),
        ("nostalgia", "melancholy", "evokes"),
    ]

    for term1, term2, rel_type in relationship_tests:
        try:
            strength = emotion_mgr.analyze_emotional_relationship(
                term1, term2, rel_type
            )
            print(f"'{term1}' {rel_type} '{term2}': {strength:.2f}")
        except Exception as e:
            print(f"Error analyzing {rel_type} between '{term1}' and '{term2}': {e}")

    print("-" * 60)


if __name__ == "__main__":
    main()
