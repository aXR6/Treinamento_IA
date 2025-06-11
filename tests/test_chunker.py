import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

# We'll patch get_sbert_model in adaptive_chunker to avoid heavy model loading
class DummyTokenizer:
    def __init__(self, max_length=512):
        self.model_max_length = max_length

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

class DummyModel:
    def __init__(self, max_length=512):
        self.max_seq_length = max_length
        self.tokenizer = DummyTokenizer(max_length)

@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    import types
    dummy_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)
    dummy_nltk = types.ModuleType("nltk")
    dummy_nltk.corpus = types.SimpleNamespace(wordnet=None)
    dummy_nltk.data = types.SimpleNamespace(find=lambda name: None)
    dummy_nltk.download = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "nltk", dummy_nltk)
    monkeypatch.setitem(sys.modules, "nltk.corpus", dummy_nltk.corpus)
    dummy_dotenv = types.ModuleType("dotenv")
    dummy_dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "dotenv", dummy_dotenv)
    monkeypatch.setitem(sys.modules, "fitz", types.ModuleType("fitz"))
    monkeypatch.setitem(sys.modules, "pikepdf", types.ModuleType("pikepdf"))
    dummy_langchain = types.ModuleType("langchain")
    dummy_langchain.text_splitter = types.ModuleType("text_splitter")
    def _simple_splitter(chunk_size=0, chunk_overlap=0):
        def split(text):
            tokens = text.split()
            out = []
            start = 0
            while start < len(tokens):
                end = start + chunk_size if chunk_size else len(tokens)
                out.append(" ".join(tokens[start:end]))
                start = end - chunk_overlap
            return out
        return types.SimpleNamespace(split_text=split)

    dummy_langchain.text_splitter.TokenTextSplitter = lambda **k: _simple_splitter(k.get("chunk_size", 0), k.get("chunk_overlap", 0))
    dummy_langchain.text_splitter.RecursiveCharacterTextSplitter = lambda **k: _simple_splitter(k.get("chunk_size", 0), k.get("chunk_overlap", 0))
    monkeypatch.setitem(sys.modules, "langchain", dummy_langchain)
    monkeypatch.setitem(sys.modules, "langchain.text_splitter", dummy_langchain.text_splitter)
    dummy_st = types.ModuleType("sentence_transformers")
    dummy_st.SentenceTransformer = lambda *a, **k: DummyModel()
    monkeypatch.setitem(sys.modules, "sentence_transformers", dummy_st)
    dummy_transformers = types.ModuleType("transformers")
    dummy_tf_utils = types.ModuleType("utils")
    dummy_tf_utils.logging = types.SimpleNamespace(set_verbosity_error=lambda: None)
    dummy_transformers.utils = dummy_tf_utils
    monkeypatch.setitem(sys.modules, "transformers", dummy_transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", dummy_tf_utils)
    import adaptive_chunker
    monkeypatch.setattr(adaptive_chunker, "get_sbert_model", lambda *a, **k: DummyModel())
    yield


def run_with_chunk_size(size):
    os.environ["CHUNK_SIZE"] = str(size)
    import config
    config.CHUNK_SIZE = size
    import adaptive_chunker
    importlib.reload(adaptive_chunker)
    adaptive_chunker.get_sbert_model = lambda *a, **k: DummyModel()
    text = " ".join([f"word{i}" for i in range(200)])
    chunks = adaptive_chunker.hierarchical_chunk(text, {})
    return [len(c.split()) for c in chunks]


def test_chunk_size_affects_length():
    lengths_50 = run_with_chunk_size(50)
    lengths_30 = run_with_chunk_size(30)
    assert max(lengths_50) > max(lengths_30)

