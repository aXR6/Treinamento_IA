import importlib
import os
import types
import sys
import contextlib

import pytest


def load_main(monkeypatch):
    env = {
        'PG_HOST': 'x', 'PG_PORT': '5432', 'PG_USER': 'u', 'PG_PASSWORD': 'p', 'PG_DB_PDF': 'd_pdf', 'PG_DB_QA': 'd_qa',
        'OLLAMA_EMBEDDING_MODEL': 'm1', 'SERAFIM_EMBEDDING_MODEL': 'm2',
        'MINILM_L6_V2': 'm3', 'MINILM_L12_V2': 'm4', 'MPNET_EMBEDDING_MODEL': 'm5',
        'DIM_MXBAI': '1', 'DIM_SERAFIM': '1', 'DIM_MINILM_L6': '1', 'DIM_MINIL12': '1', 'DIM_MPNET': '1'
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    # dummy heavy deps
    monkeypatch.setitem(sys.modules, 'fitz', types.ModuleType('fitz'))
    monkeypatch.setitem(sys.modules, 'pdfplumber', types.ModuleType('pdfplumber'))
    monkeypatch.setitem(sys.modules, 'pytesseract', types.ModuleType('pytesseract'))
    high = types.ModuleType('pdfminer.high_level')
    high.extract_text = lambda p: ''
    monkeypatch.setitem(sys.modules, 'pdfminer.high_level', high)
    monkeypatch.setitem(sys.modules, 'pdfminer', types.ModuleType('pdfminer'))
    loaders = types.ModuleType('langchain_community.document_loaders')
    loaders.PyPDFLoader = lambda *a, **k: types.SimpleNamespace(load=lambda: [])
    loaders.PDFMinerLoader = lambda *a, **k: types.SimpleNamespace(load=lambda: [])
    loaders.UnstructuredWordDocumentLoader = lambda *a, **k: types.SimpleNamespace(load=lambda: [])
    monkeypatch.setitem(sys.modules, 'langchain_community.document_loaders', loaders)
    pil = types.ModuleType('PIL'); pil.Image = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'PIL', pil)
    monkeypatch.setitem(sys.modules, 'psycopg2', types.ModuleType('psycopg2'))
    monkeypatch.setitem(sys.modules, 'torch', types.ModuleType('torch'))
    st_mod = types.ModuleType('sentence_transformers'); st_mod.CrossEncoder = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'sentence_transformers', st_mod)
    qg_mod = types.ModuleType('question_generation'); qg_mod.pipeline = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'question_generation', qg_mod)
    tf_mod = types.ModuleType('transformers'); tf_mod.pipeline = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'transformers', tf_mod)
    tqdm_mod = types.ModuleType('tqdm'); tqdm_mod.tqdm = lambda x, **k: x
    monkeypatch.setitem(sys.modules, 'tqdm', tqdm_mod)
    prom = types.ModuleType('prometheus_client')
    prom.start_http_server = lambda *a, **k: None
    prom.Counter = prom.Histogram = prom.Gauge = lambda *a, **k: types.SimpleNamespace(inc=lambda *a, **k: None, observe=lambda *a, **k: None, set=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, 'prometheus_client', prom)
    dotenv = types.ModuleType('dotenv'); dotenv.load_dotenv = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'dotenv', dotenv)

    import config
    importlib.reload(config)
    monkeypatch.setattr(config, 'validate_config', lambda: None)
    import metrics, utils
    monkeypatch.setattr(metrics, 'start_metrics_server', lambda *a, **k: None)
    monkeypatch.setattr(utils, 'setup_logging', lambda: None)
    import main
    importlib.reload(main)
    return main


def test_select_functions(monkeypatch):
    main = load_main(monkeypatch)
    monkeypatch.setattr('builtins.input', lambda prompt='': '2')
    assert main.select_strategy('pypdf') == main.STRATEGY_OPTIONS[1]
    assert main.select_embedding('m1') == main.EMBED_MODELS['2']
    assert main.select_dimension(1) == main.DIMENSIONS['2']
    assert main.select_database('d_pdf') == main.DB_OPTIONS['2']


def test_test_model(monkeypatch):
    main = load_main(monkeypatch)
    tf = sys.modules['transformers']

    class DummyTok:
        def __call__(self, text, return_tensors='pt'):
            return {'input_ids': types.SimpleNamespace(to=lambda d: [0])}
        def decode(self, ids, skip_special_tokens=True):
            return 'out'

    class DummyModel:
        def __init__(self):
            self.device = 'cpu'
            self.called = False
        def to(self, dev):
            self.device = dev
            return self
        def generate(self, **k):
            self.called = True
            return [[1]]

    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda p: DummyTok())
    dummy = DummyModel()
    tf.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda p: dummy)

    monkeypatch.setattr(main.torch, 'no_grad', lambda: contextlib.nullcontext(), raising=False)

    inputs = iter(['hi', ''])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    main.test_model('m', 'cpu')
    assert dummy.called


def test_chat_model(monkeypatch):
    main = load_main(monkeypatch)
    tf = sys.modules['transformers']

    class DummyTok:
        def __call__(self, text, return_tensors='pt'):
            return {'input_ids': types.SimpleNamespace(to=lambda d: [0])}
        def decode(self, ids, skip_special_tokens=True):
            return 'resp'

    class DummyModel:
        def __init__(self):
            self.device = 'cpu'
            self.calls = 0
        def to(self, dev):
            self.device = dev
            return self
        def generate(self, **k):
            self.calls += 1
            return [[1, 2]]

    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda p: DummyTok())
    dummy = DummyModel()
    tf.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda p: dummy)

    monkeypatch.setattr(main.torch, 'no_grad', lambda: contextlib.nullcontext(), raising=False)

    inputs = iter(['oi', 'tchau', ''])
    monkeypatch.setattr('builtins.input', lambda prompt='': next(inputs))
    main.chat_model('m', 'cpu')
    assert dummy.calls == 2

