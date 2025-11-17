import sys
from pathlib import Path
import types
import importlib

# Ensure repository source on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Stub heavy dependencies used by ParserRefiner
nltk = types.ModuleType("nltk")
corpus = types.ModuleType("nltk.corpus")
wordnet_mod = types.ModuleType("nltk.corpus.wordnet")
reader_mod = types.ModuleType("nltk.corpus.reader.wordnet")


class Lemma: ...


class Synset: ...


reader_mod.Lemma = Lemma
reader_mod.Synset = Synset
corpus.wordnet = wordnet_mod
nltk.corpus = corpus
nltk.download = lambda *a, **k: None
nltk.Tree = object
stem_mod = types.ModuleType("nltk.stem")


class WordNetLemmatizer: ...


stem_mod.WordNetLemmatizer = WordNetLemmatizer
nltk.stem = stem_mod
sys.modules["nltk"] = nltk
sys.modules["nltk.corpus"] = corpus
sys.modules["nltk.corpus.wordnet"] = wordnet_mod
sys.modules["nltk.corpus.reader"] = types.ModuleType("nltk.corpus.reader")
sys.modules["nltk.corpus.reader.wordnet"] = reader_mod
sys.modules["nltk.stem"] = stem_mod

sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules.setdefault("rdflib", types.ModuleType("rdflib"))
transformers_mod = types.ModuleType("transformers")


class Dummy: ...


transformers_mod.AutoModelForCausalLM = Dummy
transformers_mod.AutoTokenizer = Dummy
transformers_mod.PreTrainedModel = Dummy
transformers_mod.PreTrainedTokenizer = Dummy
sys.modules["transformers"] = transformers_mod


def test_cli_start_exists():
    module = importlib.import_module("word_forge.forge")
    assert hasattr(module, "start")
    assert callable(module.start)
    assert hasattr(module, "main")
    assert callable(module.main)


def test_cli_argument_parsing(monkeypatch):
    module = importlib.import_module("word_forge.forge")
    captured = {}

    def fake_start(words=None, run_minutes=None, worker_count=4):
        captured["args"] = {
            "words": words,
            "minutes": run_minutes,
            "workers": worker_count,
        }

    monkeypatch.setattr(module, "start", fake_start)
    module.main(["start", "test", "--minutes", "1", "--workers", "2"])
    assert captured["args"] == {"words": ["test"], "minutes": 1.0, "workers": 2}


def test_graph_build_command(monkeypatch):
    module = importlib.import_module("word_forge.forge")
    called = {}

    def fake_run_graph_build(**kwargs):
        called["kwargs"] = kwargs
        return True

    monkeypatch.setattr(module, "run_graph_build", fake_run_graph_build)
    result = module.main(["graph", "build", "--timeout", "5", "--poll-interval", "2"])
    assert result == 0
    assert called["kwargs"] == {"poll_interval": 2.0, "timeout": 5.0}


def test_graph_visualize_command(monkeypatch):
    module = importlib.import_module("word_forge.forge")
    called = {}

    def fake_run_graph_visualization(**kwargs):
        called["kwargs"] = kwargs
        return True

    monkeypatch.setattr(module, "run_graph_visualization", fake_run_graph_visualization)
    result = module.main(
        [
            "graph",
            "visualize",
            "--3d",
            "--open-browser",
            "--output",
            "demo.html",
        ]
    )
    assert result == 0
    assert called["kwargs"] == {
        "output_path": "demo.html",
        "use_3d": True,
        "open_in_browser": True,
    }


def test_vector_index_command(monkeypatch):
    module = importlib.import_module("word_forge.forge")
    called = {}

    def fake_run_vector_index(**kwargs):
        called["kwargs"] = kwargs
        return True

    monkeypatch.setattr(module, "run_vector_index", fake_run_vector_index)
    result = module.main(
        [
            "vector",
            "index",
            "--embedder",
            "mini",
            "--timeout",
            "10",
            "--poll-interval",
            "0.5",
        ]
    )
    assert result == 0
    assert called["kwargs"] == {
        "embedder": "mini",
        "poll_interval": 0.5,
        "timeout": 10.0,
    }


def test_emotion_annotate_command(monkeypatch):
    module = importlib.import_module("word_forge.forge")
    called = {}

    def fake_run_emotion_annotation(**kwargs):
        called["kwargs"] = kwargs
        return True

    monkeypatch.setattr(module, "run_emotion_annotation", fake_run_emotion_annotation)
    result = module.main(
        [
            "emotion",
            "annotate",
            "--strategy",
            "hybrid",
            "--timeout",
            "15",
            "--poll-interval",
            "1",
        ]
    )
    assert result == 0
    assert called["kwargs"] == {
        "strategy": "hybrid",
        "poll_interval": 1.0,
        "timeout": 15.0,
    }


def test_demo_full_command(monkeypatch):
    module = importlib.import_module("word_forge.forge")
    called = {}

    def fake_run_demo_full(**kwargs):
        called["kwargs"] = kwargs
        return True

    monkeypatch.setattr(module, "run_demo_full", fake_run_demo_full)
    result = module.main(["demo", "full", "--3d", "--open-browser", "--timeout", "20"])
    assert result == 0
    assert called["kwargs"] == {
        "use_3d": True,
        "open_in_browser": True,
        "timeout": 20.0,
    }
