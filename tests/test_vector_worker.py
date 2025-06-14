import sys
import types
import time
from pathlib import Path

sys.modules.setdefault("torch", types.ModuleType("torch"))
torch_mod = sys.modules["torch"]
setattr(torch_mod, "device", lambda *a, **k: "cpu")
setattr(torch_mod, "cuda", types.SimpleNamespace(is_available=lambda: False))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

nltk = types.ModuleType("nltk")
nltk.sentiment = types.ModuleType("nltk.sentiment")
vader_mod = types.ModuleType("nltk.sentiment.vader")
vader_mod.SentimentIntensityAnalyzer = lambda *a, **k: None
nltk.sentiment.vader = vader_mod
sys.modules["nltk"] = nltk
sys.modules["nltk.sentiment"] = nltk.sentiment
sys.modules["nltk.sentiment.vader"] = vader_mod

chromadb = types.ModuleType("chromadb")
chromadb.Client = lambda *a, **k: None
chromadb.PersistentClient = lambda *a, **k: None
sys.modules["chromadb"] = chromadb

textblob_module = types.ModuleType("textblob")
textblob_module.TextBlob = lambda *a, **k: None
sys.modules["textblob"] = textblob_module

transformers_mod = types.ModuleType("transformers")
transformers_mod.AutoModelForCausalLM = lambda *a, **k: None
transformers_mod.AutoTokenizer = lambda *a, **k: None
transformers_mod.PreTrainedModel = lambda *a, **k: None
transformers_mod.PreTrainedTokenizer = lambda *a, **k: None
transformers_mod.PreTrainedTokenizerFast = lambda *a, **k: None
transformers_mod.PretrainedConfig = lambda *a, **k: None
sys.modules["transformers"] = transformers_mod

sys.modules.setdefault("rdflib", types.ModuleType("rdflib"))

sentence_module = types.ModuleType("sentence_transformers")
sentence_module.SentenceTransformer = lambda *a, **k: None
sys.modules["sentence_transformers"] = sentence_module

import types

numpy_stub = types.ModuleType("numpy")


def _zeros(size, dtype=None):
    length = size if isinstance(size, int) else size[0]
    return [0.0] * length


numpy_stub.float32 = float
numpy_stub.zeros = _zeros
sys.modules["numpy"] = numpy_stub
numpy_typing = types.ModuleType("numpy.typing")
numpy_typing.NDArray = object
numpy_stub.typing = numpy_typing
sys.modules["numpy.typing"] = numpy_typing

import numpy as np
import pytest

from word_forge.database.database_manager import DBManager
from word_forge.vectorizer.vector_worker import VectorWorker

class DummyEmbedder:
    def embed(self, text: str):
        return np.zeros(1, dtype=np.float32)

class DummyVectorStore:
    def __init__(self):
        self.ids = []

    def upsert(self, id_: int, vector):
        self.ids.append(id_)


def test_worker_skips_unmodified_words(tmp_path):
    db = DBManager(db_path=tmp_path / "test.db")
    db.insert_or_update_word("alpha", "first")
    db.insert_or_update_word("beta", "second")

    store = DummyVectorStore()
    worker = VectorWorker(db, store, DummyEmbedder())

    words = worker._get_all_words()
    worker._process_words(words)
    worker.last_processed = time.time()

    first_ids = list(store.ids)

    time.sleep(0.01)
    db.insert_or_update_word("gamma", "third")
    db.insert_or_update_word("alpha", "updated")

    words2 = worker._get_all_words()
    terms2 = {w.term for w in words2}
    assert terms2 == {"gamma", "alpha"}

    worker._process_words(words2)
    assert store.ids[: len(first_ids)] == first_ids
    assert set(store.ids[len(first_ids):]) == {
        db.get_word_id("gamma"),
        db.get_word_id("alpha"),
    }
