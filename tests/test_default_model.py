import inspect

from fastapi.testclient import TestClient

import main


def test_openai_omitted_model_uses_server_default():
    default = inspect.signature(main.openai_transcribe).parameters['model'].default
    assert default.default == main.DEFAULT_MODEL


def test_webui_uses_configured_default(monkeypatch):
    monkeypatch.setattr(main, 'DEFAULT_MODEL', 'gigaam-v3')
    response = TestClient(main.app).get('/')
    assert 'const DEFAULT_MODEL = "gigaam-v3";' in response.text


def test_explicit_whisper_does_not_become_gigaam(monkeypatch):
    monkeypatch.setattr(main, 'DEFAULT_MODEL', 'gigaam-v3')
    monkeypatch.setattr(main.torch.cuda, 'is_available', lambda: True)
    assert main._select_openai_whisper_model({'gigaam-v3': object()}) == 'large-v3'
