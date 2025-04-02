import sqlite3
import time
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Union, cast

from textblob import TextBlob

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# Updated imports - import the class instead of constants
from word_forge.configs.emotion_config import EmotionConfig
from word_forge.database.db_manager import DBManager
from word_forge.emotion.emotion_types import (
    EmotionalContext,
    EmotionAnalysisDict,
    EmotionDimension,
    FullEmotionAnalysisDict,
    MessageEmotionDict,
    WordEmotionDict,
)


class EmotionError(Exception):
    """Exception raised for errors in the emotion analysis subsystem."""

    pass


class EmotionManager:
    """Tracks emotional associations with words and conversation messages.

    This class manages the storage and retrieval of emotional data:
    - Word emotions: valence (positive/negative) and arousal levels
    - Message emotions: categorical emotion labels with confidence scores

    All emotional data is persistent and stored in SQLite tables.

    The emotion analysis uses a hybrid approach combining TextBlob and VADER
    (when available) for more accurate sentiment detection in varied texts.

    Advanced recursive emotion processing is available for deeper analysis.
    """

    def __init__(self, db_manager: DBManager) -> None:
        """Initialize the emotion tracking system.

        Args:
            db_manager: Database manager providing connection to storage

        Raises:
            EmotionError: If there's an issue initializing the emotion tables
        """
        self.db_manager = db_manager

        # Ensure schema compatibility with migrations
        self._run_schema_migrations()

        # Create config instance
        self.config = EmotionConfig()
        self._create_tables()
        self._init_sentiment_analyzers()

        # Access emotion constraints from centralized config instance
        self.VALENCE_RANGE = self.config.valence_range
        self.AROUSAL_RANGE = self.config.arousal_range
        self.CONFIDENCE_RANGE = self.config.confidence_range
        self.EMOTION_KEYWORDS = self.config.emotion_keywords

        # Hybrid analysis weights
        self.vader_weight = self.config.vader_weight
        self.textblob_weight = self.config.textblob_weight

        # Initialize recursive emotion processor lazily
        self._recursive_processor = None

    def _run_schema_migrations(self) -> None:
        """Run necessary database schema migrations."""
        try:
            # Import here to avoid circular imports
            from word_forge.database.db_migration import SchemaMigrator

            migrator = SchemaMigrator(self.db_manager)
            migrator.migrate_all()
        except Exception as e:
            print(f"Warning: Schema migration error (continuing anyway): {e}")

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

    def _init_sentiment_analyzers(self) -> None:
        """Initialize sentiment analysis tools if available."""
        # Only initialize VADER if enabled in config
        self.vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None

    @contextmanager
    def _db_connection(self):
        """Create a database connection using the DBManager's path.

        Yields:
            sqlite3.Connection: An active database connection
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
            raise EmotionError(f"Failed to initialize emotion tables: {e}") from e

    def set_word_emotion(self, word_id: int, valence: float, arousal: float) -> None:
        """Store or update emotional values for a word."""
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
            raise EmotionError(f"Failed to store word emotion: {e}") from e

    def get_word_emotion(self, word_id: int) -> Optional[WordEmotionDict]:
        """Retrieve emotional data for a word.

        Args:
            word_id: Database ID of the target word

        Returns:
            Dictionary containing emotional data (valence, arousal, timestamp),
            or None if no data exists for the word

        Raises:
            EmotionError: If the database operation fails
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
            raise EmotionError(f"Failed to retrieve word emotion: {e}") from e

    def set_message_emotion(
        self, message_id: int, label: str, confidence: float = 1.0
    ) -> None:
        """Tag a message with an emotion label.

        Args:
            message_id: Database ID of the target message
            label: Emotional category name (e.g., 'happy', 'sad')
            confidence: Certainty level of the emotion (0.0 to 1.0)

        Raises:
            EmotionError: If the database operation fails
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
            raise EmotionError(f"Failed to store message emotion: {e}") from e

    def get_message_emotion(self, message_id: int) -> Optional[MessageEmotionDict]:
        """Retrieve emotional data for a message.

        Args:
            message_id: Database ID of the target message

        Returns:
            Dictionary containing emotion label, confidence and timestamp,
            or None if no data exists for the message

        Raises:
            EmotionError: If the database operation fails
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
            raise EmotionError(f"Failed to retrieve message emotion: {e}") from e

    def _analyze_with_vader(self, text: str) -> Tuple[float, float]:
        """Analyze text sentiment using VADER.

        VADER is specifically tuned for social media content and handles
        emoticons, slang, and capitalization particularly well.

        Args:
            text: Text to analyze for emotional content

        Returns:
            Tuple of (valence, arousal) values from VADER
        """
        if not self.vader:
            return 0.0, 0.0

        vader_scores = self.vader.polarity_scores(text)

        # Map VADER compound score (-1 to 1) directly to valence
        valence = vader_scores["compound"]

        # Estimate arousal from VADER's intensity measures
        # Higher intensity (positive + negative) suggests higher arousal
        intensity = vader_scores["pos"] + vader_scores["neg"]
        arousal = min(intensity * 1.5, 1.0)  # Scale and cap to our range

        return valence, arousal

    def analyze_text_emotion(self, text: str) -> Tuple[float, float]:
        """Analyze text to determine emotional valence and arousal.

        Uses a hybrid approach combining TextBlob and VADER (when available)
        for more accurate sentiment analysis across different text styles.

        Args:
            text: Text to analyze for emotional content

        Returns:
            Tuple of (valence, arousal) values normalized to proper ranges
        """
        # TextBlob analysis (always available)
        blob = TextBlob(text)
        textblob_valence = blob.sentiment.polarity

        # Calculate text characteristics for arousal
        subjectivity = blob.sentiment.subjectivity
        exclamation_count = text.count("!")
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        # VADER analysis (when available)
        if self.vader:
            vader_valence, vader_arousal = self._analyze_with_vader(text)

            # Weighted fusion of both analyzers using config weights
            valence = (vader_valence * self.vader_weight) + (
                textblob_valence * self.textblob_weight
            )

            # Combine VADER arousal with text characteristics
            arousal_factors = [
                vader_arousal * 2,  # VADER's arousal is weighted higher
                subjectivity,
                min(exclamation_count / 5, 1.0),
                uppercase_ratio * 2,
            ]
            arousal = min(sum(arousal_factors) / 4, 1.0)
        else:
            # Fallback to only TextBlob (original behavior)
            valence = textblob_valence

            # Original arousal calculation
            arousal_factors = [
                subjectivity,
                min(exclamation_count / 5, 1.0),
                uppercase_ratio * 2,
            ]
            arousal = min(sum(arousal_factors) / 3, 1.0)

        return valence, arousal

    def _count_keyword_occurrences(self, text: str) -> Dict[str, int]:
        """Count emotion keyword occurrences in a text.

        Args:
            text: Text to analyze for emotion keywords

        Returns:
            Dictionary mapping emotion categories to keyword occurrence counts
        """
        text_lower = text.lower()
        return {
            emotion: sum(text_lower.count(word) for word in keywords)
            for emotion, keywords in self.EMOTION_KEYWORDS.items()
        }

    def classify_emotion(self, text: str) -> Tuple[str, float]:
        """Classify text into basic emotion categories.

        Uses sentiment analysis and keyword detection to determine the most
        likely emotion expressed in the text.

        Args:
            text: Text to analyze for emotional content

        Returns:
            Tuple of (emotion_label, confidence)
        """
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

        return emotion, confidence

    def process_message(self, message_id: int, text: str) -> EmotionAnalysisDict:
        """Process a message to identify and store its emotional content.

        Args:
            message_id: Database ID of the message
            text: Message text to analyze

        Returns:
            Dictionary containing the identified emotion data

        Raises:
            EmotionError: If storing the emotion data fails
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

        This method provides a more comprehensive emotional analysis compared
        to basic sentiment analysis, including meta-emotions and patterns.

        Args:
            term: The word or phrase to analyze
            context: Optional emotional context or context name

        Returns:
            Dictionary containing comprehensive emotional analysis

        Note:
            This is a more computationally intensive analysis than basic
            sentiment classification and should be used selectively.
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
        primary_dims = concept.primary_emotion.dimensions
        valence = primary_dims.get(EmotionDimension.VALENCE, 0.0)
        arousal = primary_dims.get(EmotionDimension.AROUSAL, 0.0)
        emotion_label, confidence = self.classify_emotion(term)

        # Create the return structure
        result = {
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

        return cast(FullEmotionAnalysisDict, result)

    def analyze_emotional_relationship(
        self, term1: str, term2: str, relationship_type: str
    ) -> float:
        """Analyze the emotional relationship between two terms.

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
        - meta_emotion: Emotion about an emotion
        """
        return self.recursive_processor.analyze_relationship(
            term1, term2, relationship_type
        )

    def create_emotional_context(
        self,
        domain: str = None,
        cultural_factors: Dict[str, float] = None,
        situational_factors: Dict[str, float] = None,
        temporal_factors: Dict[str, float] = None,
        domain_specific: Dict[str, float] = None,
    ) -> EmotionalContext:
        """Create a custom emotional context for analysis.

        Args:
            domain: Optional domain name for predefined contexts
            cultural_factors: Optional cultural dimension adjustments
            situational_factors: Optional situational adjustments
            temporal_factors: Optional temporal dimension adjustments
            domain_specific: Optional domain-specific adjustments

        Returns:
            An emotional context object for use in recursive analysis
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

        Args:
            name: Name to identify this context
            context: The emotional context to register
        """
        self.recursive_processor.register_context(name, context)

    def get_emotional_context(self, name: str) -> Optional[EmotionalContext]:
        """Retrieve a registered emotional context by name.

        Args:
            name: Name of the context to retrieve

        Returns:
            The emotional context if found, None otherwise
        """
        return self.recursive_processor.get_context(name)


def main() -> None:
    """Demonstrate EmotionManager functionality with comprehensive emotion analysis."""

    # Initialize with actual DBManager instead of temporary database
    db_path = "word_forge.sqlite"
    db_manager = DBManager(db_path=db_path)

    # Create database tables if they don't exist
    db_manager._create_tables()

    # Initialize emotion manager
    emotion_mgr = EmotionManager(db_manager)

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

    # Process and track emotions for words in the database
    try:
        # Get all words from the database
        all_words = db_manager.get_all_words()
        word_count = len(all_words)
        print(f"\nAnalyzing emotions for {word_count} words in the database:")
        print("-" * 60)

        # Process a sample of words (max 5) for demonstration
        sample_size = min(100, word_count)
        for i, word_data in enumerate(all_words[:sample_size]):
            word_id = word_data["id"]
            term = word_data["term"]
            definition = word_data["definition"] or ""

            # Extract usage examples
            usage_examples = []
            if word_data["usage_examples"]:
                usage_str = word_data["usage_examples"]
                usage_examples = usage_str.split("; ") if usage_str else []

            # Analyze the word itself
            word_valence, word_arousal = emotion_mgr.analyze_text_emotion(term)

            # Analyze the definition if available
            if definition:
                def_valence, def_arousal = emotion_mgr.analyze_text_emotion(definition)
                def_emotion, def_confidence = emotion_mgr.classify_emotion(definition)

            # Store emotion for the word
            emotion_mgr.set_word_emotion(word_id, word_valence, word_arousal)

            # Display results
            print(f"Word {i+1}/{sample_size}: '{term}'")
            print(
                f"  - Word Emotion: Valence={word_valence:.2f}, Arousal={word_arousal:.2f}"
            )

            if definition:
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

            # Verify storage
            stored_emotion = emotion_mgr.get_word_emotion(word_id)
            print(f"  - Stored in DB: {stored_emotion}")
            print("-" * 60)

        if word_count > sample_size:
            print(
                f"...and {word_count - sample_size} more words (showing {sample_size} for brevity)"
            )

    except Exception as e:
        print(f"Error processing word emotions: {e}")

    # Example: Analyze and track emotions for messages
    messages = [
        (100, "I'm so happy today! Everything is going GREAT!"),
        (101, "I feel sad and disappointed about the results."),
        (102, "This product is amazing and exceeds expectations."),
        (103, "I'm furious about the terrible customer service."),
        (104, "Just received the package, it looks fine."),
        (105, "OMG this is AWESOME!!! 😊 I love it so much!"),
    ]

    # Display analyzer availability
    analyzer_status = "✓ VADER" if VADER_AVAILABLE else "✗ VADER (using TextBlob only)"
    print(f"\nEmotion Analyzers: TextBlob + {analyzer_status}")

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

    # Demonstrate recursive emotion analysis
    print("\nRecursive Emotion Analysis:")
    print("-" * 60)
    sample_terms = ["happiness", "melancholy", "excitement", "anxiety", "curiosity"]

    for term in sample_terms:
        print(f"Analyzing '{term}' recursively:")
        try:
            deep_analysis = emotion_mgr.analyze_term_recursively(term)
            print(
                f"  Basic emotion: {deep_analysis['emotion_label']} (confidence: {deep_analysis['confidence']:.2f})"
            )
            print(f"  Recursive depth: {deep_analysis['recursive_depth']}")
            print(f"  Primary dimensions: {deep_analysis['dimensions']}")

            # Show meta-emotions if available
            if deep_analysis.get("meta_emotions"):
                print("  Meta-emotions:")
                for meta in deep_analysis["meta_emotions"][
                    :2
                ]:  # Show first 2 for brevity
                    print(f"    - {meta['label']}")

            # Show patterns if available
            if deep_analysis.get("patterns"):
                print("  Emotional patterns:")
                for pattern_type in deep_analysis["patterns"]:
                    print(
                        f"    - {pattern_type}: {len(deep_analysis['patterns'][pattern_type])} states"
                    )

            print("-" * 60)
        except Exception as e:
            print(f"  Error in recursive analysis: {e}")
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
