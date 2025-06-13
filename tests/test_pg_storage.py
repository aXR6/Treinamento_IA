import types
import contextlib
import importlib
import sys

import pytest


@pytest.fixture()
def pg(monkeypatch):
    dummy_prom = types.ModuleType('prometheus_client')
    class _M:
        def inc(self):
            pass
        def observe(self, *a):
            pass
        def set(self, *a):
            pass
    dummy_prom.start_http_server = lambda *a, **k: None
    dummy_prom.Counter = lambda *a, **k: _M()
    dummy_prom.Histogram = lambda *a, **k: _M()
    dummy_prom.Gauge = lambda *a, **k: _M()
    monkeypatch.setitem(sys.modules, 'prometheus_client', dummy_prom)
    dummy_psy = types.ModuleType('psycopg2')
    dummy_psy.connect = lambda **k: None
    monkeypatch.setitem(sys.modules, 'psycopg2', dummy_psy)
    dummy_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
        no_grad=lambda: contextlib.nullcontext()
    )
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch)
    st_mod = types.ModuleType('sentence_transformers')
    class DummyCE:
        def __init__(self, *a, **k):
            pass
        def predict(self, pairs):
            return [0.5 for _ in pairs]
    st_mod.CrossEncoder = DummyCE
    monkeypatch.setitem(sys.modules, 'sentence_transformers', st_mod)
    qg_mod = types.ModuleType('question_generation'); qg_mod.pipeline = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'question_generation', qg_mod)
    tf_mod = types.ModuleType('transformers'); tf_mod.pipeline = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'transformers', tf_mod)
    import pg_storage
    importlib.reload(pg_storage)
    return pg_storage


class DummyTorch:
    def __init__(self):
        self.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)
    def no_grad(self):
        return contextlib.nullcontext()


def test_generate_embedding_pad(pg, monkeypatch):
    dummy_torch = DummyTorch()
    monkeypatch.setattr(pg, 'torch', dummy_torch)
    model = types.SimpleNamespace(encode=lambda text, convert_to_numpy=True: [1, 2])
    monkeypatch.setattr(pg, 'get_sbert_model', lambda n, device: model)
    result = pg.generate_embedding('txt', 'm', 5, 'cuda')
    assert result == [1, 2, 0.0, 0.0, 0.0]


def test_generate_embedding_fallback(pg, monkeypatch):
    dummy_torch = DummyTorch()
    monkeypatch.setattr(pg, 'torch', dummy_torch)
    def get_model(name, device):
        if device == 'cuda':
            return types.SimpleNamespace(encode=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('out of memory')))
        return types.SimpleNamespace(encode=lambda *a, **k: [0.1, 0.2, 0.3])
    monkeypatch.setattr(pg, 'get_sbert_model', get_model)
    result = pg.generate_embedding('txt', 'm', 3, 'cuda')
    assert result == [0.1, 0.2, 0.3]


class DummyConn:
    def __init__(self):
        self.cur = DummyCursor()
    def cursor(self):
        return self.cur
    def commit(self):
        pass
    def rollback(self):
        pass
    def close(self):
        pass

class DummyCursor:
    def __init__(self):
        self.calls = []
        self.i = 1
    def execute(self, q, params):
        self.calls.append((q, params))
    def fetchone(self):
        val = self.i
        self.i += 1
        return [val]

class DummyCE:
    def predict(self, pairs):
        return [0.5 for _ in pairs]


def test_save_to_postgres(pg, monkeypatch):
    monkeypatch.setattr(pg.psycopg2, 'connect', lambda **k: DummyConn())
    monkeypatch.setattr(pg, 'hierarchical_chunk_generator', lambda t,m,*a,**k: ['c1','c2'])
    monkeypatch.setattr(pg, 'generate_embedding', lambda text, m, d, device: [0.0,0.0,0.0])
    monkeypatch.setattr(pg, 'generate_qa', lambda text: ('Q','A'))
    monkeypatch.setattr(pg, 'get_cross_encoder', lambda model, device: DummyCE())
    res = pg.save_to_postgres('f','txt', {'__query':'q'}, 'm', 3, 'cpu')
    assert len(res) == 2
    assert res[0]['id'] == 1
    assert res[0]['question'] == 'Q'
    assert res[0]['answer'] == 'A'
    assert 'rerank_score' in res[0]


def test_generate_qa_limits_doc_stride(pg, monkeypatch):
    class DummyQA:
        def __init__(self):
            self.calls = []
            self.tokenizer = types.SimpleNamespace(
                model_max_length=16,
                tokenize=lambda text: text.split(),
                num_special_tokens_to_add=lambda pair=True: 2,
            )

        def __call__(self, question, context, **kwargs):
            self.calls.append(kwargs)
            return {"answer": "A"}

    qa = DummyQA()
    monkeypatch.setattr(pg, "_QG_AVAILABLE", True)
    monkeypatch.setattr(pg, "_QG_PIPELINE", lambda text: ["Q"])
    monkeypatch.setattr(pg, "_QA_PIPELINE", qa)
    monkeypatch.setattr(pg, "MAX_SEQ_LENGTH", 32)

    q, a = pg.generate_qa("word " * 40)
    assert q and a
    kwargs = qa.calls[0]
    specials = qa.tokenizer.num_special_tokens_to_add(pair=True)
    available = qa.tokenizer.model_max_length - specials - len(q.split())
    assert kwargs["doc_stride"] < available
    assert kwargs["max_seq_len"] <= qa.tokenizer.model_max_length


def test_generate_qa_explicit_prompt_fallback(pg, monkeypatch):
    class DummyQA:
        def __init__(self):
            self.calls = []
            self.tokenizer = types.SimpleNamespace(
                model_max_length=16,
                tokenize=lambda text: text.split(),
                num_special_tokens_to_add=lambda pair=True: 2,
            )
            self.model = types.SimpleNamespace()  # sem metodo generate

        def __call__(self, question, context, **kwargs):
            self.calls.append(kwargs)
            return {"answer": "A"}

    qa = DummyQA()
    monkeypatch.setattr(pg, "_QG_AVAILABLE", True)
    monkeypatch.setattr(pg, "_QG_PIPELINE", lambda text: ["Q"])
    monkeypatch.setattr(pg, "_QA_PIPELINE", qa)
    monkeypatch.setattr(pg, "MAX_SEQ_LENGTH", 32)
    monkeypatch.setattr(pg, "QA_EXPLICIT_PROMPT", True)

    q, a = pg.generate_qa("word " * 40)
    assert q == "Q"
    assert a == "A"
    assert len(qa.calls) == 1

