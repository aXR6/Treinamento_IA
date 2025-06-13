import importlib
import os
import types
import sys

import pytest


@pytest.fixture()
def training_module(monkeypatch):
    dummy_psy = types.ModuleType('psycopg2')
    monkeypatch.setitem(sys.modules, 'psycopg2', dummy_psy)

    dummy_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, 'torch', dummy_torch)

    tf = types.ModuleType('transformers')
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: object())
    tf.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *a, **k: object())
    tf.Trainer = object
    tf.TrainingArguments = object
    tf.DataCollatorForLanguageModeling = object
    tf.TrainerCallback = object
    monkeypatch.setitem(sys.modules, 'transformers', tf)

    tq = types.ModuleType('tqdm')
    tq.tqdm = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'tqdm', tq)

    ds = types.ModuleType('datasets')
    def from_generator(gen, features=None):
        gen = gen()
        next(gen)
        return []
    ds.Dataset = types.SimpleNamespace(from_generator=from_generator)
    ds.Features = lambda *a, **k: None
    ds.Value = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, 'datasets', ds)

    import training
    importlib.reload(training)
    return training


def test_should_use_cuda_gpu(monkeypatch, training_module):
    monkeypatch.setattr(training_module.torch.cuda, 'is_available', lambda: True)
    assert training_module._should_use_cuda('gpu', False) is True
    monkeypatch.setattr(training_module.torch.cuda, 'is_available', lambda: False)
    assert training_module._should_use_cuda('gpu', True) is False


def test_should_use_cuda_auto(monkeypatch, training_module):
    monkeypatch.setattr(training_module.torch.cuda, 'is_available', lambda: True)
    assert training_module._should_use_cuda('auto', True) is True
    monkeypatch.setattr(training_module.torch.cuda, 'is_available', lambda: False)
    assert training_module._should_use_cuda('auto', True) is False
    monkeypatch.setattr(training_module.torch.cuda, 'is_available', lambda: True)
    assert training_module._should_use_cuda('auto', False) is False


def test_should_use_cuda_cpu_skips_check(monkeypatch, training_module):
    def fail():
        raise AssertionError('cuda check called')
    monkeypatch.setattr(training_module.torch.cuda, 'is_available', fail)
    assert training_module._should_use_cuda('cpu', True) is False


def test_resolve_device(monkeypatch, training_module):
    monkeypatch.setattr(training_module.torch.cuda, 'is_available', lambda: True)
    assert training_module._resolve_device('gpu', False) == 'cuda'
    assert training_module._resolve_device('auto', True) == 'cuda'
    monkeypatch.setattr(training_module.torch.cuda, 'is_available', lambda: False)
    assert training_module._resolve_device('gpu', True) == 'cpu'
    assert training_module._resolve_device('auto', True) == 'cpu'
    assert training_module._resolve_device('auto', False) == 'cpu'
    assert training_module._resolve_device('cpu', True) == 'cpu'


def test_env_restored_on_exception_train_model(monkeypatch, training_module):
    monkeypatch.setenv('TRANSFORMERS_NO_CUDA', 'orig')
    monkeypatch.setattr(training_module, '_fetch_texts', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('fail')))

    training_module.train_model(1, 'cpu')

    assert os.environ.get('TRANSFORMERS_NO_CUDA') == 'orig'


def test_env_restored_on_exception_train_qa_model(monkeypatch, training_module):
    monkeypatch.setenv('TRANSFORMERS_NO_CUDA', 'orig')
    monkeypatch.setattr(training_module, '_fetch_qa_pairs', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('fail')))

    training_module.train_qa_model(1, 'cpu')

    assert os.environ.get('TRANSFORMERS_NO_CUDA') == 'orig'
