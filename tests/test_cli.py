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
