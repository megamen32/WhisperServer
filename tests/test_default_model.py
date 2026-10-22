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
    """A loaded GigaAM model is never auto-selected for the Whisper alias.

    The auto-selection helper only considers the Whisper family, so a GigaAM
    model that happens to be resident must not capture the alias. The
    configured default is pinned here so the check holds under every
    deployment, including the Russian-first one.
    """
    monkeypatch.setattr(main, 'DEFAULT_MODEL', 'gigaam-v3')
    monkeypatch.setattr(main, 'OPENAI_DEFAULT_MODEL', 'large-v3')
    monkeypatch.setattr(main.torch.cuda, 'is_available', lambda: True)
    assert main._select_openai_whisper_model({'gigaam-v3': object()}) == 'large-v3'
