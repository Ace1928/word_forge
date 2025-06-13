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
    def encode(self, *a, **k):
        import numpy as np
        return np.zeros(5, dtype=np.float32)
sentence_module.SentenceTransformer = DummyModel
sys.modules["sentence_transformers"] = sentence_module

emotion_module = types.ModuleType("word_forge.emotion.emotion_manager")
class DummyEmotionManager:
    def process_message(self, message_id, text):
        pass
emotion_module.EmotionManager = DummyEmotionManager
sys.modules["word_forge.emotion.emotion_manager"] = emotion_module

from types import SimpleNamespace

import pytest

from word_forge.conversation.conversation_manager import ConversationManager
from word_forge.configs.config_essentials import Result
from word_forge.database.database_manager import DBManager


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

    db_path = tmp_path / "conv.db"
    dbm = DBManager(db_path=db_path)
    return ConversationManager(
        db_manager=dbm,
        emotion_manager=StubEmotionManager(),
        graph_manager=StubGraphManager(),
        vector_store=SimpleNamespace(),
        reflexive_model=StubModel(),
        lightweight_model=StubModel(),
        affective_model=StubModel(),
        identity_model=StubModel(),
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
