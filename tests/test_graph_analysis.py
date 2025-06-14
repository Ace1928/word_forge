from pathlib import Path
import sys
import types

# Stub dependencies before importing project modules
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

numpy_stub = types.ModuleType("numpy")


def _zeros(size, dtype=None):
    length = size if isinstance(size, int) else size[0]
    return [0.0] * length


def _allclose(a, b, **_):
    return a == b


numpy_stub.float32 = float
numpy_stub.zeros = _zeros
numpy_stub.allclose = _allclose

sys.modules["numpy"] = numpy_stub
numpy_typing = types.ModuleType("numpy.typing")
numpy_typing.NDArray = object
numpy_stub.typing = numpy_typing
sys.modules["numpy.typing"] = numpy_typing

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from word_forge.graph.graph_manager import GraphManager
from word_forge.database.database_manager import DBManager


def create_manager(tmp_path):
    db = DBManager(tmp_path / "test.db")
    return GraphManager(db_manager=db)


def test_emotional_subgraph_context_filtering(tmp_path):
    manager = create_manager(tmp_path)
    a = manager.add_word_node("happy", {"valence": 0.8})
    b = manager.add_word_node("joyful", {"valence": 0.9})
    c = manager.add_word_node("sad", {"valence": -0.6})

    manager.add_relationship(a, b, "joy_associated", dimension="emotional", weight=1.0)
    manager.add_relationship(
        a, c, "sadness_associated", dimension="emotional", weight=1.0
    )

    manager.integrate_emotional_context(
        "clinical", {"sadness_associated": 0.2, "joy_associated": 1.0}
    )

    sub = manager.get_emotional_subgraph(
        "happy", depth=1, context="clinical", min_intensity=0.5
    )

    terms = {manager.query.get_term_by_id(n) for n in sub.nodes}
    assert "sad" not in terms
    assert "joyful" in terms
