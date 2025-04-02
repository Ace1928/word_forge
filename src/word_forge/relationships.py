# Expanded relationship types to include all relationships from ParserRefiner
RELATIONSHIP_TYPES = {
    # Core relationships
    "synonym": {"weight": 1.0, "color": "#4287f5", "bidirectional": True},
    "antonym": {"weight": 0.9, "color": "#f54242", "bidirectional": True},
    # Hierarchical relationships
    "hypernym": {"weight": 0.7, "color": "#42f584", "bidirectional": False},
    "hyponym": {"weight": 0.7, "color": "#a142f5", "bidirectional": False},
    # Part-whole relationships
    "holonym": {"weight": 0.6, "color": "#f5a142", "bidirectional": False},
    "meronym": {"weight": 0.6, "color": "#42f5f5", "bidirectional": False},
    # Translation relationships
    "translation": {"weight": 0.8, "color": "#42d4f5", "bidirectional": True},
    # Semantic field relationships
    "domain": {"weight": 0.5, "color": "#7a42f5", "bidirectional": False},
    "function": {"weight": 0.5, "color": "#f542a7", "bidirectional": False},
    # General semantic relationships
    "related": {"weight": 0.4, "color": "#42f5a1", "bidirectional": True},
    # Derivational relationships
    "derived_from": {"weight": 0.5, "color": "#8c42f5", "bidirectional": False},
    "etymological_source": {"weight": 0.4, "color": "#f5b942", "bidirectional": False},
    # Usage relationships
    "context": {"weight": 0.3, "color": "#42d4f5", "bidirectional": False},
    "register": {"weight": 0.3, "color": "#f542d4", "bidirectional": False},
    # Example-based relationships
    "example_of": {"weight": 0.3, "color": "#7adbf5", "bidirectional": False},
    "instance": {"weight": 0.4, "color": "#e642f5", "bidirectional": False},
    # Emotional valence relationships
    "emotional_synonym": {"weight": 0.85, "color": "#8a2be2", "bidirectional": True},
    "emotional_antonym": {"weight": 0.85, "color": "#ff6347", "bidirectional": True},
    # Emotional intensity relationships
    "intensifies": {"weight": 0.70, "color": "#ff4500", "bidirectional": False},
    "diminishes": {"weight": 0.70, "color": "#4682b4", "bidirectional": False},
    # Emotional causality relationships
    "evokes": {"weight": 0.65, "color": "#da70d6", "bidirectional": False},
    "responds_to": {"weight": 0.65, "color": "#32cd32", "bidirectional": False},
    # Emotional dimension relationships
    "valence_related": {"weight": 0.60, "color": "#ff69b4", "bidirectional": True},
    "arousal_related": {"weight": 0.60, "color": "#ff8c00", "bidirectional": True},
    "dominance_related": {"weight": 0.60, "color": "#4b0082", "bidirectional": True},
    # Emotional complexity relationships
    "emotional_component": {"weight": 0.55, "color": "#7b68ee", "bidirectional": False},
    "emotional_composite": {"weight": 0.55, "color": "#ee82ee", "bidirectional": False},
    "emotional_sequence": {"weight": 0.50, "color": "#6a5acd", "bidirectional": False},
    # Contextual emotional relationships
    "cultural_context": {"weight": 0.45, "color": "#daa520", "bidirectional": False},
    "situational_context": {"weight": 0.45, "color": "#20b2aa", "bidirectional": False},
    "temporal_context": {"weight": 0.45, "color": "#778899", "bidirectional": False},
    # Meta-emotional relationships
    "meta_emotion": {"weight": 0.40, "color": "#800080", "bidirectional": False},
    "emotional_regulation": {
        "weight": 0.40,
        "color": "#008080",
        "bidirectional": False,
    },
    # Default for any other relationship
    "default": {"weight": 0.3, "color": "#aaaaaa", "bidirectional": True},
}
