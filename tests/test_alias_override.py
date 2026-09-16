"""Run with OPENAI_DEFAULT_MODEL=gigaam-v3 to check deployed alias routing."""
import asyncio
import os
import json
from types import SimpleNamespace

import pytest
import main

pytestmark = pytest.mark.skipif(os.getenv('OPENAI_DEFAULT_MODEL') != 'gigaam-v3', reason='requires GigaAM alias configuration')


@pytest.mark.parametrize('endpoint', ['native', 'openai'])
def test_whisper_alias_routes_to_gigaam(monkeypatch, endpoint):
    async def run():
        monkeypatch.setattr(main, '_get_cached_result', lambda *a: None)
        monkeypatch.setattr(main, '_require_openai_api_key', lambda *a: None)
        monkeypatch.setattr(main, 'pending_results', {})
        class Upload:
            filename = 'voice.wav'
            async def read(self):
                return b'audio'
        def put(request):
            assert request['model'] == 'gigaam-v3'
            assert request['requested_model'] == 'whisper-1'
            main.pending_results.pop(request['request_id']).set_result({'text': 'готово'})
        monkeypatch.setattr(main.app.state, 'request_queue', SimpleNamespace(put=put), raising=False)
        if endpoint == 'native':
            result = await main._transcribe_impl(Upload(), 'whisper-1', None, None, 0, False, False, True)
        else:
            result = await main.openai_transcribe(None, Upload(), 'whisper-1', None, 0, 'json', False, True, False)
        if hasattr(result, 'body'):
            result = json.loads(result.body)
        assert result['text'] == 'готово'
    asyncio.run(run())


def test_alias_cache_changes_with_backend(monkeypatch):
    monkeypatch.setitem(main.OPENAI_MODEL_MAP, 'whisper-1', main.OPENAI_WHISPER_INTERNAL_MODEL)
    old = main.transcription_cache_key('native', 'whisper-1', None, 'hash', vad_filter=True)
    monkeypatch.setitem(main.OPENAI_MODEL_MAP, 'whisper-1', 'gigaam-v3')
    new = main.transcription_cache_key('native', 'whisper-1', None, 'hash', vad_filter=True)
    assert new != old
