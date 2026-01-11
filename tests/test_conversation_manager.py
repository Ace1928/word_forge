import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import types

# Stub networkx before importing project modules
networkx_stub = types.ModuleType("networkx")


class _Graph:
    def __init__(self):
        self._nodes = {}
        self._edges = {}

    def add_node(self, node, **attrs):
        self._nodes.setdefault(node, {}).update(attrs)

    def add_edge(self, u, v, **attrs):
        self.add_node(u)
        self.add_node(v)
        self._edges[(u, v)] = attrs

    def nodes(self, data=False):
        return list(self._nodes.items()) if data else list(self._nodes.keys())

    def edges(self, data=False):
        return (
            [(*k, v) for k, v in self._edges.items()]
            if data
            else list(self._edges.keys())
        )

    def number_of_nodes(self):
        return len(self._nodes)

    def number_of_edges(self):
        return len(self._edges)

    def clear(self):
        self._nodes.clear()
        self._edges.clear()

    def subgraph(self, nodes):
        g = _Graph()
        nset = set(nodes)
        for n in nset:
            if n in self._nodes:
                g._nodes[n] = self._nodes[n].copy()
        for (u, v), d in self._edges.items():
            if u in nset and v in nset:
                g._edges[(u, v)] = d.copy()
        return g

    def degree(self):
        deg = {n: 0 for n in self._nodes}
        for u, v in self._edges:
            deg[u] += 1
            deg[v] += 1
        return [(n, deg[n]) for n in self._nodes]

    def get_edge_data(self, u, v):
        return self._edges.get((u, v), self._edges.get((v, u), {}))

    def copy(self):
        g = _Graph()
        g._nodes = {k: v.copy() for k, v in self._nodes.items()}
        g._edges = {(u, v): d.copy() for (u, v), d in self._edges.items()}
        return g


def _layout_stub(G, dim=2, **_):
    return {n: (float(i),) * dim for i, n in enumerate(G.nodes())}


def _ego_graph(G, node, radius=1):
    seen = {node}
    frontier = {node}
    for _ in range(radius):
        nxt = set()
        for u in frontier:
            for a, b in G.edges():
                if a == u and b not in seen:
                    nxt.add(b)
                    seen.add(b)
                if b == u and a not in seen:
                    nxt.add(a)
                    seen.add(a)
        frontier = nxt
    return G.subgraph(seen)


networkx_stub.Graph = _Graph
networkx_stub.DiGraph = _Graph
networkx_stub.spring_layout = _layout_stub
networkx_stub.spectral_layout = _layout_stub
networkx_stub.circular_layout = _layout_stub
networkx_stub.ego_graph = _ego_graph

nx_agraph = types.ModuleType("networkx.nx_agraph")
nx_agraph.graphviz_layout = _layout_stub
networkx_stub.nx_agraph = nx_agraph

alg_mod = types.ModuleType("networkx.algorithms")
alg_comm = types.ModuleType("networkx.algorithms.community")
alg_comm.louvain_communities = lambda *a, **k: []
alg_mod.community = alg_comm
networkx_stub.algorithms = alg_mod


def set_node_attributes(G, attr_dict):
    for node, attrs in attr_dict.items():
        if node in G._nodes:
            G._nodes[node].update(attrs)


networkx_stub.set_node_attributes = set_node_attributes

sys.modules["networkx"] = networkx_stub
sys.modules["networkx.algorithms"] = alg_mod
sys.modules["networkx.algorithms.community"] = alg_comm
sys.modules["networkx.nx_agraph"] = nx_agraph

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
