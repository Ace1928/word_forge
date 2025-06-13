import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import types
chromadb = types.ModuleType("chromadb")
chromadb.Client = lambda *a, **k: None
chromadb.PersistentClient = lambda *a, **k: None
sys.modules["chromadb"] = chromadb
sentence_module = types.ModuleType("sentence_transformers")
class DummyModel:
    def __init__(self, *a, **k):
        pass
    def get_sentence_embedding_dimension(self):
        return 5
    def encode(self, text, normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False):
        import numpy as np
        return np.zeros(5, dtype=np.float32)
sentence_module.SentenceTransformer = DummyModel
sys.modules["sentence_transformers"] = sentence_module

import numpy as np
import pytest

from word_forge.vectorizer.vector_store import VectorStore, DimensionMismatchError, SearchError


def test_validate_vector_dimension_mismatch():
    vs = object.__new__(VectorStore)
    vs.dimension = 4
    with pytest.raises(DimensionMismatchError):
        vs._validate_vector_dimension(np.zeros(3, dtype=np.float32))


def test_search_requires_input():
    vs = object.__new__(VectorStore)
    with pytest.raises(SearchError):
        vs.search()
