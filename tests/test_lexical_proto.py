import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import types
import importlib

# Stub heavy dependencies before importing lexical_proto
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
sys.modules["nltk"] = nltk
sys.modules["nltk.corpus"] = corpus
sys.modules["nltk.corpus.wordnet"] = wordnet_mod
sys.modules["nltk.corpus.reader"] = types.ModuleType("nltk.corpus.reader")
sys.modules["nltk.corpus.reader.wordnet"] = reader_mod

sys.modules["torch"] = types.ModuleType("torch")
rdflib_mod = types.ModuleType("rdflib")


class Graph: ...


class Literal: ...


class URIRef: ...


rdflib_mod.Graph = Graph
rdflib_mod.Literal = Literal
rdflib_mod.URIRef = URIRef
sys.modules["rdflib"] = rdflib_mod
transformers_mod = types.ModuleType("transformers")


class Dummy: ...


transformers_mod.AutoModelForCausalLM = Dummy
transformers_mod.AutoTokenizer = Dummy
transformers_mod.PreTrainedModel = Dummy
transformers_mod.PreTrainedTokenizer = Dummy
transformers_mod.PreTrainedTokenizerFast = Dummy
sys.modules["transformers"] = transformers_mod
transformers_mod.generation = types.ModuleType("transformers.generation")
utils_mod = types.ModuleType("transformers.generation.utils")


class GenerateBeamDecoderOnlyOutput: ...


class GenerateBeamEncoderDecoderOutput: ...


class GenerateDecoderOnlyOutput: ...


class GenerateEncoderDecoderOutput: ...


utils_mod.GenerateBeamDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput
utils_mod.GenerateBeamEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput
utils_mod.GenerateDecoderOnlyOutput = GenerateDecoderOnlyOutput
utils_mod.GenerateEncoderDecoderOutput = GenerateEncoderDecoderOutput
transformers_mod.generation.utils = utils_mod
sys.modules["transformers.generation"] = transformers_mod.generation
sys.modules["transformers.generation.utils"] = utils_mod

lexical_proto = importlib.import_module("lexical_proto")
from lexical_proto import (
    file_exists,
    safely_open_file,
    read_json_file,
    read_jsonl_file,
)



def test_file_exists(tmp_path):
    p = tmp_path / "t.txt"
    assert not file_exists(p)
    p.write_text("data")
    assert file_exists(p)


def test_safely_open_file_missing(tmp_path):
    assert safely_open_file(tmp_path / "no.txt") is None


def test_read_json_file_invalid(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json")
    assert read_json_file(p, {"d": 1}) == {"d": 1}


def test_read_jsonl_file(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"a":1}\ninvalid\n{"a":2}\n')

    def proc(d):
        return d["a"]

    assert read_jsonl_file(p, proc) == [1, 2]


def test_read_jsonl_file_missing(tmp_path):
    p = tmp_path / "missing.jsonl"
    assert read_jsonl_file(p, lambda d: d) == []
