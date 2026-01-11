"""Shared pytest fixtures for Word Forge test suite.

This module provides common fixtures, mocks, and utilities used across
the test suite. It includes:
- Database fixtures for isolated testing
- Mock fixtures for heavy dependencies (torch, chromadb, transformers)
- Configuration fixtures for consistent test environments
- Temporary path fixtures for file-based tests
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is in path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest.fixture
def db_manager(tmp_path: Path):
    """Create an isolated DBManager instance for testing.

    Returns:
        DBManager: A fresh database manager with tables created
    """
    from word_forge.database.database_manager import DBManager

    db_path = tmp_path / "test.db"
    manager = DBManager(db_path=db_path)
    manager.create_tables()
    return manager


@pytest.fixture
def populated_db_manager(db_manager):
    """Create a DBManager with sample data.

    Returns:
        DBManager: Database manager with test words and relationships
    """
    # Add sample words
    db_manager.insert_or_update_word("happiness", "a state of joy", "noun")
    db_manager.insert_or_update_word("sadness", "a state of sorrow", "noun")
    db_manager.insert_or_update_word("joy", "intense happiness", "noun")
    db_manager.insert_or_update_word("sorrow", "deep sadness", "noun")
    db_manager.insert_or_update_word("anger", "strong displeasure", "noun")
    db_manager.insert_or_update_word("love", "deep affection", "noun")
    db_manager.insert_or_update_word("fear", "emotional response to danger", "noun")
    db_manager.insert_or_update_word("surprise", "unexpected emotion", "noun")

    # Add relationships
    db_manager.insert_relationship("happiness", "joy", "synonym")
    db_manager.insert_relationship("happiness", "sadness", "antonym")
    db_manager.insert_relationship("sadness", "sorrow", "synonym")
    db_manager.insert_relationship("joy", "happiness", "synonym")
    db_manager.insert_relationship("love", "happiness", "evokes")

    return db_manager


@pytest.fixture
def emotion_manager(db_manager):
    """Create an EmotionManager instance for testing.

    Returns:
        EmotionManager: Manager configured with test database
    """
    from word_forge.emotion.emotion_manager import EmotionManager

    return EmotionManager(db_manager)


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def test_config():
    """Create a test configuration.

    Returns:
        dict: Configuration dictionary for testing
    """
    return {
        "database": {
            "db_path": ":memory:",
            "pool_size": 1,
        },
        "vectorizer": {
            "model_name": "test-model",
            "batch_size": 8,
        },
        "queue": {
            "max_workers": 2,
            "queue_size": 100,
        },
        "logging": {
            "level": "DEBUG",
        },
    }


# =============================================================================
# Mock Fixtures for Heavy Dependencies
# =============================================================================


@pytest.fixture
def mock_torch():
    """Mock the torch library for testing without GPU dependencies.

    Yields:
        MagicMock: Mocked torch module
    """
    mock = MagicMock()
    mock.cuda.is_available.return_value = False
    mock.device.return_value = "cpu"

    with patch.dict("sys.modules", {"torch": mock}):
        yield mock


@pytest.fixture
def mock_transformers():
    """Mock the transformers library.

    Yields:
        MagicMock: Mocked transformers module
    """
    mock = MagicMock()
    mock.AutoTokenizer.from_pretrained.return_value = MagicMock()
    mock.AutoModel.from_pretrained.return_value = MagicMock()

    with patch.dict("sys.modules", {"transformers": mock}):
        yield mock


@pytest.fixture
def mock_sentence_transformers():
    """Mock the sentence-transformers library.

    Yields:
        MagicMock: Mocked sentence_transformers module
    """
    mock = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 384]  # Standard embedding size
    mock.SentenceTransformer.return_value = mock_model

    with patch.dict("sys.modules", {"sentence_transformers": mock}):
        yield mock


@pytest.fixture
def mock_chromadb():
    """Mock the ChromaDB library.

    Yields:
        MagicMock: Mocked chromadb module
    """
    mock = MagicMock()
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "distances": [[0.1, 0.2]],
        "documents": [["doc1", "doc2"]],
    }
    mock_collection.add.return_value = None
    mock_collection.count.return_value = 10

    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock.Client.return_value = mock_client
    mock.PersistentClient.return_value = mock_client

    with patch.dict("sys.modules", {"chromadb": mock}):
        yield mock


@pytest.fixture
def mock_faiss():
    """Mock the FAISS library.

    Yields:
        MagicMock: Mocked faiss module
    """
    mock = MagicMock()
    mock_index = MagicMock()
    mock_index.ntotal = 100
    mock_index.search.return_value = (
        [[0.1, 0.2]],  # distances
        [[0, 1]],  # indices
    )
    mock.IndexFlatL2.return_value = mock_index
    mock.IndexIVFFlat.return_value = mock_index

    with patch.dict("sys.modules", {"faiss": mock}):
        yield mock


@pytest.fixture
def mock_spacy():
    """Mock the spaCy library.

    Yields:
        MagicMock: Mocked spacy module
    """
    mock = MagicMock()
    mock_doc = MagicMock()
    mock_doc.has_vector = True
    mock_doc.vector = [0.1] * 96
    mock_doc.similarity.return_value = 0.8

    mock_nlp = MagicMock()
    mock_nlp.return_value = mock_doc

    mock.load.return_value = mock_nlp

    with patch.dict("sys.modules", {"spacy": mock}):
        yield mock


@pytest.fixture
def mock_nltk():
    """Mock the NLTK library.

    Yields:
        MagicMock: Mocked nltk module
    """
    mock = MagicMock()
    mock.corpus.wordnet.synsets.return_value = []
    mock.data.find.return_value = "/fake/path"

    with patch.dict("sys.modules", {"nltk": mock, "nltk.corpus": mock.corpus}):
        yield mock


@pytest.fixture
def mock_all_heavy_deps(
    mock_torch, mock_transformers, mock_sentence_transformers, mock_chromadb
):
    """Mock all heavy dependencies at once.

    This fixture combines all heavy dependency mocks for tests that need
    to avoid loading any ML libraries.

    Yields:
        dict: Dictionary containing all mocked modules
    """
    yield {
        "torch": mock_torch,
        "transformers": mock_transformers,
        "sentence_transformers": mock_sentence_transformers,
        "chromadb": mock_chromadb,
    }


# =============================================================================
# Helper Fixtures
# =============================================================================


@pytest.fixture
def sample_emotion_vector():
    """Create a sample emotion vector for testing.

    Returns:
        EmotionVector: A test emotion vector
    """
    from word_forge.emotion.emotion_types import EmotionDimension, EmotionVector

    return EmotionVector(
        dimensions={
            EmotionDimension.VALENCE: 0.7,
            EmotionDimension.AROUSAL: 0.5,
            EmotionDimension.DOMINANCE: 0.3,
        },
        confidence=0.85,
    )


@pytest.fixture
def sample_emotional_context():
    """Create a sample emotional context for testing.

    Returns:
        EmotionalContext: A test emotional context
    """
    from word_forge.emotion.emotion_types import EmotionalContext

    context = EmotionalContext()
    context.domain_specific = {"valence": 0.2, "arousal": -0.1}
    context.cultural_factors = {"intensity_modifier": 0.9}
    return context


@pytest.fixture
def sample_graph():
    """Create a sample NetworkX graph for testing.

    Returns:
        nx.DiGraph: A test graph with sample nodes and edges
    """
    import networkx as nx

    G = nx.DiGraph()
    G.add_node("happiness", emotion_valence=0.8, emotion_arousal=0.6)
    G.add_node("sadness", emotion_valence=-0.8, emotion_arousal=0.3)
    G.add_node("joy", emotion_valence=0.9, emotion_arousal=0.7)
    G.add_edge("happiness", "joy", relationship="synonym", weight=0.9)
    G.add_edge("happiness", "sadness", relationship="antonym", weight=0.8)
    return G


# =============================================================================
# Markers and Skip Conditions
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "requires_nltk: marks tests that require NLTK data"
    )
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")


def _check_nltk_available() -> bool:
    """Check if NLTK WordNet data is available."""
    try:
        import nltk

        nltk.data.find("corpora/wordnet")
        return True
    except (ImportError, LookupError):
        return False


def _check_spacy_available() -> bool:
    """Check if spaCy model is available."""
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except (ImportError, OSError):
        return False


# Skip conditions - defined after helper functions
requires_nltk = pytest.mark.skipif(
    not _check_nltk_available(), reason="NLTK data not available"
)

requires_spacy = pytest.mark.skipif(
    not _check_spacy_available(), reason="spaCy model not available"
)
