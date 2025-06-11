import importlib
import types
import sys

import pytest


@pytest.fixture()
def extractors_module(monkeypatch):
    dummy = types.ModuleType('fitz')
    monkeypatch.setitem(sys.modules, 'fitz', dummy)
    monkeypatch.setitem(sys.modules, 'pdfplumber', types.ModuleType('pdfplumber'))
    monkeypatch.setitem(sys.modules, 'pytesseract', types.ModuleType('pytesseract'))
    pdfminer_high = types.ModuleType('pdfminer.high_level')
    pdfminer_high.extract_text = lambda p: ''
    monkeypatch.setitem(sys.modules, 'pdfminer.high_level', pdfminer_high)
    monkeypatch.setitem(sys.modules, 'pdfminer', types.ModuleType('pdfminer'))
    loaders = types.ModuleType('langchain_community.document_loaders')
    loaders.PyPDFLoader = lambda *a, **k: types.SimpleNamespace(load=lambda: [])
    loaders.PDFMinerLoader = lambda *a, **k: types.SimpleNamespace(load=lambda: [])
    loaders.UnstructuredWordDocumentLoader = lambda *a, **k: types.SimpleNamespace(load=lambda: [])
    monkeypatch.setitem(sys.modules, 'langchain_community.document_loaders', loaders)
    pil_module = types.ModuleType('PIL')
    pil_module.Image = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'PIL', pil_module)
    import extractors
    importlib.reload(extractors)
    return extractors


class DummyStrategy:
    def __init__(self, name):
        self.name = name
        self.calls = []
    def extract(self, path):
        self.calls.append(path)
        return f"text from {self.name}"


def test_docx_uses_unstructured(extractors_module, monkeypatch):
    dummy = DummyStrategy('unstructured')
    monkeypatch.setitem(extractors_module.STRATEGIES_MAP, 'unstructured', dummy)
    result = extractors_module.extract_text('file.docx', 'pypdf')
    assert result == 'text from unstructured'
    assert dummy.calls == ['file.docx']


def test_image_uses_image_strategy(extractors_module, monkeypatch):
    dummy = DummyStrategy('image')
    monkeypatch.setitem(extractors_module.STRATEGIES_MAP, 'image', dummy)
    result = extractors_module.extract_text('photo.png', 'pdfminer')
    assert result == 'text from image'
    assert dummy.calls == ['photo.png']


def test_pdf_fallback_to_pdfminer(extractors_module, monkeypatch):
    primary = DummyStrategy('pypdf')
    primary.extract = lambda path: ''
    monkeypatch.setitem(extractors_module.STRATEGIES_MAP, 'pypdf', primary)
    monkeypatch.setattr(extractors_module, 'repair_pdf', lambda p: p)
    miner_text = 'x' * (extractors_module.OCR_THRESHOLD + 1)
    monkeypatch.setattr(extractors_module, 'pdfminer_extract_text', lambda p: miner_text)
    result = extractors_module.extract_text('file.pdf', 'pypdf')
    assert result == miner_text

