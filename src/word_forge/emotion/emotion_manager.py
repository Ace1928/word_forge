import sqlite3
import time
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Union

from textblob import TextBlob

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

from word_forge.database.db_manager import DBManager


class EmotionManager:
    """Tracks emotional associations with words and conversation messages.

    This class manages the storage and retrieval of emotional data:
    - Word emotions: valence (positive/negative) and arousal levels
    - Message emotions: categorical emotion labels with confidence scores

    All emotional data is persistent and stored in SQLite tables.

    The emotion analysis uses a hybrid approach combining TextBlob and VADER
    (when available) for more accurate sentiment detection in varied texts.
    """

    # Emotion value constraints
    VALENCE_RANGE = (-1.0, 1.0)  # Negative to positive
    AROUSAL_RANGE = (0.0, 1.0)  # Calm to excited
    CONFIDENCE_RANGE = (0.0, 1.0)  # Certainty level

    # Emotion categories with their keywords for classification
    EMOTION_KEYWORDS = {
        "happiness": ["happy", "joy", "delight", "pleased", "glad", "excited"],
        "sadness": ["sad", "unhappy", "depressed", "down", "miserable", "gloomy"],
        "anger": ["angry", "furious", "enraged", "mad", "irritated", "annoyed"],
        "fear": ["afraid", "scared", "frightened", "terrified", "anxious", "worried"],
        "surprise": ["surprised", "astonished", "amazed", "shocked", "startled"],
        "disgust": ["disgusted", "revolted", "repulsed", "sickened", "appalled"],
        "neutral": ["okay", "fine", "neutral", "indifferent", "average"],
    }

    def __init__(self, db_manager: DBManager) -> None:
        """Initialize the emotion tracking system.

        Args:
            db_manager: Database manager providing connection to storage
        """
        self.db_manager = db_manager
        self._create_tables()
        self._init_sentiment_analyzers()

    def _init_sentiment_analyzers(self) -> None:
        """Initialize sentiment analysis tools if available."""
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
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            # Create word_emotion table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS word_emotion (
                    word_id INTEGER PRIMARY KEY,
                    valence REAL NOT NULL,
                    arousal REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY(word_id) REFERENCES words(id)
                )
            """
            )

            # Create message_emotion table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS message_emotion (
                    message_id INTEGER PRIMARY KEY,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """
            )
            conn.commit()

    def set_word_emotion(self, word_id: int, valence: float, arousal: float) -> None:
        """Store or update emotional values for a word.

        Args:
            word_id: Database ID of the target word
            valence: Positivity/negativity score (-1.0 to 1.0)
            arousal: Excitement/calmness level (0.0 to 1.0)

        Valence represents emotional positivity:
        - Negative values indicate negative emotions
        - Positive values indicate positive emotions

        Arousal represents emotional intensity:
        - Low values indicate calm emotions
        - High values indicate excited emotions
        """
        valence = max(self.VALENCE_RANGE[0], min(self.VALENCE_RANGE[1], valence))
        arousal = max(self.AROUSAL_RANGE[0], min(self.AROUSAL_RANGE[1], arousal))

        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO word_emotion
                (word_id, valence, arousal, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (word_id, valence, arousal, time.time()),
            )
            conn.commit()

    def get_word_emotion(self, word_id: int) -> Optional[Dict[str, Union[float, int]]]:
        """Retrieve emotional data for a word.

        Args:
            word_id: Database ID of the target word

        Returns:
            Dictionary containing emotional data (valence, arousal, timestamp),
            or None if no data exists for the word
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT word_id, valence, arousal, timestamp FROM word_emotion WHERE word_id = ?",
                (word_id,),
            )
            row = cursor.fetchone()

            if row:
                return {
                    "word_id": row["word_id"],
                    "valence": row["valence"],
                    "arousal": row["arousal"],
                    "timestamp": row["timestamp"],
                }
            return None

    def set_message_emotion(
        self, message_id: int, label: str, confidence: float = 1.0
    ) -> None:
        """Tag a message with an emotion label.

        Args:
            message_id: Database ID of the target message
            label: Emotional category name (e.g., 'happy', 'sad')
            confidence: Certainty level of the emotion (0.0 to 1.0)
        """
        confidence = max(
            self.CONFIDENCE_RANGE[0], min(self.CONFIDENCE_RANGE[1], confidence)
        )

        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO message_emotion
                (message_id, label, confidence, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (message_id, label, confidence, time.time()),
            )
            conn.commit()

    def get_message_emotion(
        self, message_id: int
    ) -> Optional[Dict[str, Union[str, float, int]]]:
        """Retrieve emotional data for a message.

        Args:
            message_id: Database ID of the target message

        Returns:
            Dictionary containing emotion label, confidence and timestamp,
            or None if no data exists for the message
        """
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT message_id, label, confidence, timestamp FROM message_emotion WHERE message_id = ?",
                (message_id,),
            )
            row = cursor.fetchone()

            if row:
                return {
                    "message_id": row["message_id"],
                    "label": row["label"],
                    "confidence": row["confidence"],
                    "timestamp": row["timestamp"],
                }
            return None

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

            # Weighted fusion of both analyzers
            # VADER gets higher weight as it's typically better for emotional expressions
            valence = (vader_valence * 0.7) + (textblob_valence * 0.3)

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
            else 0.3
        )
        confidence = 0.4 + (0.6 * keyword_strength)

        return emotion, confidence

    def process_message(
        self, message_id: int, text: str
    ) -> Dict[str, Union[str, float]]:
        """Process a message to identify and store its emotional content.

        Args:
            message_id: Database ID of the message
            text: Message text to analyze

        Returns:
            Dictionary containing the identified emotion data
        """
        # Classify the emotion in the text
        emotion_label, confidence = self.classify_emotion(text)

        # Store the emotion data
        self.set_message_emotion(message_id, emotion_label, confidence)

        return {"emotion_label": emotion_label, "confidence": confidence}


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


if __name__ == "__main__":
    main()
