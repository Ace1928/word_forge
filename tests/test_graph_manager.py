import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from word_forge.graph.graph_manager import GraphManager
from word_forge.database.database_manager import DBManager


def test_build_graph_from_db(tmp_path):
    db = DBManager(db_path=tmp_path / "test.db")
    db.insert_or_update_word("alpha", "first")
    db.insert_or_update_word("beta", "second")
    db.insert_relationship("alpha", "beta", "synonym")

    manager = GraphManager(db_manager=db)
    manager.build_graph()

    terms = {data["term"] for _, data in manager.g.nodes(data=True)}
    assert {"alpha", "beta"} <= terms
    assert manager.g.number_of_edges() == 1
