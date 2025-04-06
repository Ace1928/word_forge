import enum
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, TypedDict

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class WordEmotionDict(TypedDict):
    """Type definition for word emotion data structure."""

    word_id: int
    valence: float
    arousal: float
    timestamp: float


class MessageEmotionDict(TypedDict):
    """Type definition for message emotion data structure."""

    message_id: int
    label: str
    confidence: float
    timestamp: float


class EmotionAnalysisDict(TypedDict):
    """Type definition for emotion analysis results."""

    emotion_label: str
    confidence: float


class FullEmotionAnalysisDict(EmotionAnalysisDict, total=False):
    """Extended type definition including recursive emotion analysis."""

    concept: Dict[str, Any]
    dimensions: Dict[str, float]
    meta_emotions: List[Dict[str, Any]]
    patterns: Dict[str, List[Dict[str, Any]]]
    recursive_depth: int


class VaderSentimentScores(TypedDict):
    """Type definition for VADER sentiment analyzer output.

    The VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon
    provides sentiment intensity scores across four dimensions:

    Attributes:
        neg: Negative sentiment intensity score [0.0-1.0]
        neu: Neutral sentiment intensity score [0.0-1.0]
        pos: Positive sentiment intensity score [0.0-1.0]
        compound: Normalized compound score [-1.0-1.0], representing overall sentiment
    """

    neg: float  # Negative sentiment intensity
    neu: float  # Neutral sentiment intensity
    pos: float  # Positive sentiment intensity
    compound: float  # Normalized, weighted composite score


class EmotionDimension(enum.Enum):
    """Core emotional dimensions based on psychological models."""

    VALENCE = "valence"  # Positive vs. negative affect
    AROUSAL = "arousal"  # Intensity of emotional activation
    DOMINANCE = "dominance"  # Sense of control vs. submission
    CERTAINTY = "certainty"  # Confidence in emotional assessment
    RELEVANCE = "relevance"  # Personal significance
    NOVELTY = "novelty"  # Unexpectedness or familiarity
    AGENCY = "agency"  # Attribution of causality
    SOCIAL = "social"  # Social connection or isolation
    TEMPORAL = "temporal"  # Time orientation (past/present/future)

    # Extended PAD model dimensions
    POTENCY = "potency"  # Strength/weakness dimension
    FAMILIARITY = "familiarity"  # Knownnness/unknownness
    INTENSITY = "intensity"  # Raw strength of the emotion

    # Meta dimensions
    META_COMPLEXITY = "meta_complexity"  # How complex the emotion is
    META_STABILITY = "meta_stability"  # How stable the emotion is over time
    META_CONGRUENCE = "meta_congruence"  # How congruent with other emotions

    @classmethod
    def primary_dimensions(cls) -> Set["EmotionDimension"]:
        """Return the three primary PAD model dimensions."""
        return {cls.VALENCE, cls.AROUSAL, cls.DOMINANCE}

    @classmethod
    def extended_dimensions(cls) -> Set["EmotionDimension"]:
        """Return extended PAD model dimensions."""
        return cls.primary_dimensions() | {cls.POTENCY, cls.FAMILIARITY, cls.INTENSITY}

    @classmethod
    def meta_dimensions(cls) -> Set["EmotionDimension"]:
        """Return meta-emotional dimensions."""
        return {cls.META_COMPLEXITY, cls.META_STABILITY, cls.META_CONGRUENCE}

    @classmethod
    def all_dimensions(cls) -> Set["EmotionDimension"]:
        """Return all available emotional dimensions."""
        return {member for member in cls}

    # Define a TypedDict for VADER sentiment scores
    class VaderSentimentScores(TypedDict):
        """Type definition for VADER sentiment analyzer output."""

        pos: float
        neg: float
        neu: float
        compound: float


@dataclass(frozen=True)
class EmotionVector:
    """Immutable n-dimensional vector representing emotional state."""

    dimensions: Dict[EmotionDimension, float]
    confidence: float = 1.0

    def __post_init__(self):
        """Validate dimension values are within range."""
        for dim, value in self.dimensions.items():
            if not -1.0 <= value <= 1.0:
                object.__setattr__(
                    self,
                    "dimensions",
                    {**self.dimensions, dim: max(-1.0, min(1.0, value))},
                )

        if not 0.0 <= self.confidence <= 1.0:
            object.__setattr__(self, "confidence", max(0.0, min(1.0, self.confidence)))

    def distance(self, other: "EmotionVector") -> float:
        """Calculate the n-dimensional distance between emotion vectors."""
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys())
        return math.sqrt(
            sum(
                (self.dimensions.get(dim, 0.0) - other.dimensions.get(dim, 0.0)) ** 2
                for dim in all_dims
            )
        )

    def blend(self, other: "EmotionVector", weight: float = 0.5) -> "EmotionVector":
        """Create a weighted blend of two emotion vectors."""
        all_dims = set(self.dimensions.keys()) | set(other.dimensions.keys())
        new_dims = {
            dim: self.dimensions.get(dim, 0.0) * (1 - weight)
            + other.dimensions.get(dim, 0.0) * weight
            for dim in all_dims
        }
        new_confidence = self.confidence * (1 - weight) + other.confidence * weight
        return EmotionVector(dimensions=new_dims, confidence=new_confidence)

    def inverse(self) -> "EmotionVector":
        """Return the emotional inverse (opposite in primary dimensions)."""
        return EmotionVector(
            dimensions={
                dim: -val
                for dim, val in self.dimensions.items()
                if dim in EmotionDimension.primary_dimensions()
            },
            confidence=self.confidence,
        )

    def intensify(self, factor: float = 1.5) -> "EmotionVector":
        """Intensify the emotion by the given factor."""
        return EmotionVector(
            dimensions={dim: val * factor for dim, val in self.dimensions.items()},
            confidence=self.confidence,
        )

    def diminish(self, factor: float = 0.5) -> "EmotionVector":
        """Diminish the emotion by the given factor."""
        return EmotionVector(
            dimensions={dim: val * factor for dim, val in self.dimensions.items()},
            confidence=self.confidence,
        )

    def normalized(self) -> "EmotionVector":
        """Return a normalized vector with magnitude 1.0."""
        magnitude_squared = sum(v * v for v in self.dimensions.values())
        if magnitude_squared == 0:
            return self

        magnitude = math.sqrt(magnitude_squared)
        return EmotionVector(
            dimensions={k: v / magnitude for k, v in self.dimensions.items()},
            confidence=self.confidence,
        )

    def dominant_dimension(self) -> Optional[Tuple[EmotionDimension, float]]:
        """Return the dimension with the largest absolute value."""
        if not self.dimensions:
            return None
        return max(self.dimensions.items(), key=lambda item: abs(item[1]))

    def as_dict(self) -> Dict[str, float]:
        """Convert to standard dictionary with string keys."""
        return {dim.value: val for dim, val in self.dimensions.items()}


@dataclass
class EmotionalContext:
    """Context that influences emotional processing."""

    cultural_factors: Dict[str, float] = field(default_factory=dict)
    situational_factors: Dict[str, float] = field(default_factory=dict)
    temporal_factors: Dict[str, float] = field(default_factory=dict)
    domain_specific: Dict[str, float] = field(default_factory=dict)
    relationship_factors: Dict[str, float] = field(default_factory=dict)

    def apply_to_vector(self, vector: EmotionVector) -> EmotionVector:
        """Modify emotion vector based on contextual factors."""
        context_adjustment = {}

        # Apply cultural adjustments
        for factor_name, factor_value in self.cultural_factors.items():
            try:
                dim = EmotionDimension(factor_name)
                context_adjustment[dim] = factor_value
            except ValueError:
                pass  # Not a valid dimension name

        # Apply situation-specific adjustments
        for factor_name, factor_value in self.situational_factors.items():
            try:
                dim = EmotionDimension(factor_name)
                context_adjustment[dim] = (
                    context_adjustment.get(dim, 0.0) + factor_value
                )
            except ValueError:
                pass

        # Apply temporal adjustments
        for factor_name, factor_value in self.temporal_factors.items():
            try:
                dim = EmotionDimension(factor_name)
                context_adjustment[dim] = (
                    context_adjustment.get(dim, 0.0) + factor_value
                )
            except ValueError:
                pass

        # Apply domain-specific adjustments
        for factor_name, factor_value in self.domain_specific.items():
            try:
                dim = EmotionDimension(factor_name)
                context_adjustment[dim] = (
                    context_adjustment.get(dim, 0.0) + factor_value
                )
            except ValueError:
                pass

        # Create adjustment vector
        if not context_adjustment:
            return vector

        adjustment_vector = EmotionVector(
            dimensions={dim: adj for dim, adj in context_adjustment.items()},
            confidence=0.8,  # Context application reduces confidence slightly
        )

        # Blend original vector with contextual adjustments
        return vector.blend(adjustment_vector, weight=0.3)

    def combine(
        self, other: "EmotionalContext", weight: float = 0.5
    ) -> "EmotionalContext":
        """Combine two contexts with weighted blending."""
        result = EmotionalContext()

        # Combined dictionaries helper function
        def combine_dicts(dict1, dict2):
            result = dict1.copy()
            for k, v in dict2.items():
                if k in result:
                    result[k] = result[k] * (1 - weight) + v * weight
                else:
                    result[k] = v * weight
            return result

        # Combine all factor dictionaries
        result.cultural_factors = combine_dicts(
            self.cultural_factors, other.cultural_factors
        )
        result.situational_factors = combine_dicts(
            self.situational_factors, other.situational_factors
        )
        result.temporal_factors = combine_dicts(
            self.temporal_factors, other.temporal_factors
        )
        result.domain_specific = combine_dicts(
            self.domain_specific, other.domain_specific
        )
        result.relationship_factors = combine_dicts(
            self.relationship_factors, other.relationship_factors
        )

        return result


@dataclass
class EmotionalConcept:
    """Full emotional representation of a lexical concept."""

    term: str
    word_id: int
    primary_emotion: EmotionVector
    secondary_emotions: List[Tuple[str, EmotionVector]] = field(default_factory=list)
    meta_emotions: List[Tuple[str, EmotionVector]] = field(default_factory=list)
    emotional_patterns: Dict[str, List[EmotionVector]] = field(default_factory=dict)
    relationship_context: Dict[str, List[Tuple[str, float]]] = field(
        default_factory=dict
    )

    @property
    def recursive_depth(self) -> int:
        """Determine the recursive emotional depth."""
        return 1 + (1 if self.meta_emotions else 0) + len(self.emotional_patterns)

    def add_meta_emotion(self, label: str, emotion: EmotionVector) -> None:
        """Add an emotion about an emotion (meta-emotion)."""
        self.meta_emotions.append((label, emotion))

    def add_emotional_pattern(
        self, pattern_type: str, sequence: List[EmotionVector]
    ) -> None:
        """Add a temporal or conditional pattern of emotions."""
        self.emotional_patterns[pattern_type] = sequence

    def dominant_emotion(self) -> EmotionVector:
        """Return the most relevant emotion for this concept."""
        if not self.secondary_emotions:
            return self.primary_emotion

        # Return the strongest emotion based on intensity (arousal)
        all_emotions = [self.primary_emotion] + [e for _, e in self.secondary_emotions]
        return max(
            all_emotions,
            key=lambda e: abs(e.dimensions.get(EmotionDimension.AROUSAL, 0.0)),
        )

    def add_related_context(
        self, relationship_type: str, term: str, strength: float
    ) -> None:
        """Add a related term's contextual influence."""
        if relationship_type not in self.relationship_context:
            self.relationship_context[relationship_type] = []
        self.relationship_context[relationship_type].append((term, strength))

    def as_dict(self) -> Dict[str, any]:
        """Convert to dictionary representation for serialization."""
        return {
            "term": self.term,
            "word_id": self.word_id,
            "primary_emotion": {
                "dimensions": self.primary_emotion.as_dict(),
                "confidence": self.primary_emotion.confidence,
            },
            "secondary_emotions": [
                {
                    "label": label,
                    "dimensions": emotion.as_dict(),
                    "confidence": emotion.confidence,
                }
                for label, emotion in self.secondary_emotions
            ],
            "meta_emotions": [
                {
                    "label": label,
                    "dimensions": emotion.as_dict(),
                    "confidence": emotion.confidence,
                }
                for label, emotion in self.meta_emotions
            ],
            "emotional_patterns": {
                pattern_type: [
                    {"dimensions": emotion.as_dict(), "confidence": emotion.confidence}
                    for emotion in sequence
                ]
                for pattern_type, sequence in self.emotional_patterns.items()
            },
            "relationship_context": self.relationship_context,
            "recursive_depth": self.recursive_depth,
        }


class EmotionProcessor(Protocol):
    """Protocol defining the interface for emotion processors."""

    def process_term(
        self, term: str, context: Optional[EmotionalContext] = None
    ) -> EmotionalConcept: ...
    def analyze_relationship(
        self, term1: str, term2: str, relationship_type: str
    ) -> float: ...
    def get_emotion_vector(self, term: str) -> EmotionVector: ...
