"""Tests for ConversationManager functionality.

This module tests conversation management without stubbing networkx
to avoid polluting other tests.
"""

import sys
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Stub chromadb and other heavy dependencies (but NOT networkx)
chromadb = types.ModuleType("chromadb")
chromadb.Client = lambda *a, **k: None
chromadb.PersistentClient = lambda *a, **k: None
sys.modules["chromadb"] = chromadb
sentence_module = types.ModuleType("sentence_transformers")

nltk = types.ModuleType("nltk")
nltk.corpus = types.SimpleNamespace(stopwords=types.SimpleNamespace(words=lambda x: []))
nltk.Tree = lambda *a, **k: None
nltk.word_tokenize = lambda t: t.split()
nltk.pos_tag = lambda tokens: [(t, "NN") for t in tokens]
nltk.download = lambda *a, **k: None
nltk.corpus.wordnet = types.ModuleType("nltk.corpus.wordnet")
nltk.corpus.reader = types.ModuleType("nltk.corpus.reader")
nltk.corpus.reader.wordnet = types.ModuleType("nltk.corpus.reader.wordnet")
nltk.corpus.reader.wordnet.Lemma = type("Lemma", (), {})
nltk.corpus.reader.wordnet.Synset = type("Synset", (), {})
nltk.corpus.reader.wordnet.WordNetError = Exception
nltk.corpus.wordnet.Lemma = nltk.corpus.reader.wordnet.Lemma
nltk.corpus.wordnet.Synset = nltk.corpus.reader.wordnet.Synset
nltk.stem = types.ModuleType("nltk.stem")
nltk.stem.WordNetLemmatizer = lambda *a, **k: None
sys.modules["nltk"] = nltk
sys.modules["nltk.corpus"] = nltk.corpus
sys.modules["nltk.corpus.wordnet"] = nltk.corpus.wordnet
sys.modules["nltk.corpus.reader"] = nltk.corpus.reader
sys.modules["nltk.corpus.reader.wordnet"] = nltk.corpus.reader.wordnet
sys.modules["nltk.stem"] = nltk.stem

import numpy as np


class DummyModel:
    def __init__(self, *a, **k):
        pass

    def get_sentence_embedding_dimension(self):
        return 5

    def encode(self, *a, **k):
        return np.zeros(5, dtype=np.float32)


sentence_module.SentenceTransformer = DummyModel
sys.modules["sentence_transformers"] = sentence_module

torch_mod = types.ModuleType("torch")
torch_mod.device = lambda *a, **k: "cpu"
torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"] = torch_mod
transformers_mod = types.ModuleType("transformers")


class DummyConfig: ...


transformers_mod.AutoModelForCausalLM = DummyModel
transformers_mod.AutoTokenizer = DummyModel
transformers_mod.PreTrainedModel = DummyModel
transformers_mod.PreTrainedTokenizer = DummyModel
transformers_mod.PreTrainedTokenizerFast = DummyModel
transformers_mod.PretrainedConfig = DummyConfig
sys.modules["transformers"] = transformers_mod

rdflib_mod = types.ModuleType("rdflib")


class Graph: ...


class Literal: ...


class URIRef: ...


rdflib_mod.Graph = Graph
rdflib_mod.Literal = Literal
rdflib_mod.URIRef = URIRef
rdflib_mod.query = types.ModuleType("rdflib.query")


class ResultRow(tuple):
    pass


rdflib_mod.query.ResultRow = ResultRow
sys.modules["rdflib"] = rdflib_mod
sys.modules["rdflib.query"] = rdflib_mod.query

from types import SimpleNamespace

import pytest

from word_forge.conversation.conversation_manager import ConversationManager
from word_forge.configs.config_essentials import Result
from word_forge.database.database_manager import DBManager
from word_forge.queue.queue_manager import QueueManager


class StubModel:
    def __init__(self, return_value=None):
        self.return_value = return_value

    def generate_reflex(self, context):
        return Result.success(context)

    def process(self, context):
        return Result.success(context)

    def generate_core_response(self, context):
        context["intermediate_response"] = self.return_value or "ok"
        return Result.success(context)

    def refine_response(self, context):
        return Result.success(self.return_value or "ok")


class StubEmotionManager:
    def process_message(self, message_id, text):
        pass


class StubGraphManager:
    """Lightweight stand-in for :class:`GraphManager` used in tests."""

    pass


def create_manager(tmp_path, monkeypatch):
    """Instantiate :class:`ConversationManager` with stub dependencies.

    The real :class:`GraphManager` is patched to ``StubGraphManager`` to avoid
    heavy initialization during unit tests while leaving other tests untouched.
    """

    import word_forge.conversation.conversation_manager as cm_module

    monkeypatch.setattr(cm_module, "GraphManager", StubGraphManager, raising=False)
    import word_forge.utils.nltk_utils as nltk_utils

    monkeypatch.setattr(nltk_utils, "ensure_nltk_data", lambda: None)
    import word_forge.parser.parser_refiner as pr

    monkeypatch.setattr(pr, "ensure_nltk_data", lambda: None, raising=False)

    class StubExtractor:
        def extract_terms(self, definition, examples, original_term):
            tokens = definition.split()
            return [t.lower() for t in tokens], []

    monkeypatch.setattr(pr, "TermExtractor", StubExtractor)
    monkeypatch.setattr(cm_module, "TermExtractor", StubExtractor, raising=False)

    db_path = tmp_path / "conv.db"
    dbm = DBManager(db_path=db_path)
    queue_manager = QueueManager[str]()
    return ConversationManager(
        db_manager=dbm,
        emotion_manager=StubEmotionManager(),
        graph_manager=StubGraphManager(),
        vector_store=SimpleNamespace(),
        reflexive_model=StubModel(),
        lightweight_model=StubModel(),
        affective_model=StubModel(),
        identity_model=StubModel(),
        queue_manager=queue_manager,
    )


def test_conversation_flow(tmp_path, monkeypatch):
    cm = create_manager(tmp_path, monkeypatch)
    conv_id = cm.start_conversation().unwrap()
    add_res = cm.add_message(conv_id, "User", "Hello", generate_response=False)
    assert add_res.is_success
    conv = cm.get_conversation(conv_id).unwrap()
    assert len(conv["messages"]) == 1
    assert conv["messages"][0]["text"] == "Hello"
    end_res = cm.end_conversation(conv_id)
    assert end_res.is_success


def test_end_nonexistent_conversation(tmp_path, monkeypatch):
    cm = create_manager(tmp_path, monkeypatch)
    res = cm.end_conversation(999)
    assert res.is_failure
    assert res.error and res.error.code == "CONVERSATION_NOT_FOUND"


def test_add_message_empty(tmp_path, monkeypatch):
    cm = create_manager(tmp_path, monkeypatch)
    conv_id = cm.start_conversation().unwrap()
    with pytest.raises(ValueError):
        cm.add_message(conv_id, "User", " ")
