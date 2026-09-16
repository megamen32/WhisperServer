"""Regression: streamed audio must not collide with a later fallback request."""
import asyncio
import queue
from types import SimpleNamespace

import pytest

import main


@pytest.mark.parametrize('endpoint', ['native', 'openai'])
def test_requests_get_unique_ids_even_when_audio_object_is_reused(monkeypatch, endpoint):
    async def run():
        audio = b'RIFF identical audio retained by upload/fallback'
        class Upload:
            filename = 'voice.wav'
            async def read(self):
                return audio
        requests = []
        monkeypatch.setattr(main, 'pending_streams', {})
        monkeypatch.setattr(main, '_get_cached_result', lambda *a: None)
        monkeypatch.setattr(main, '_require_openai_api_key', lambda *a: None)
        monkeypatch.setattr(main.app.state, 'request_queue', SimpleNamespace(put=requests.append), raising=False)
        for _ in range(2):
            if endpoint == 'native':
                await main._transcribe_impl(Upload(), 'whisper-1', None, None, 0.0, True, False, True)
            else:
                await main.openai_transcribe(None, Upload(), 'whisper-1', None, 0.0, 'json', True, True, False)
        assert len({r['request_id'] for r in requests}) == 2
        assert len(main.pending_streams) == 2
    asyncio.run(run())


@pytest.mark.parametrize('stale_kind', ['cancelled', 'segment'])
def test_listener_survives_stale_response_and_delivers_next_request(monkeypatch, stale_kind):
    async def run():
        loop = asyncio.get_running_loop()
        stale = loop.create_future()
        good = loop.create_future()
        if stale_kind == 'cancelled':
            stale.cancel()
            first = {'request_id': 'old', 'result': {'text': 'old'}, 'final': True}
        else:
            first = {'request_id': 'old', 'segment': {'text': 'late segment'}}
        monkeypatch.setattr(main, 'pending_results', {'old': stale, 'good': good})
        monkeypatch.setattr(main, 'pending_streams', {})
        monkeypatch.setattr(main.app.state, 'executor', None, raising=False)
        responses = queue.Queue()
        for result in [first, {'request_id': 'good', 'result': {'text': 'delivered'}, 'final': True}, None]:
            responses.put(result)
        await asyncio.wait_for(main.response_listener(None, responses), 2)
        assert good.result() == {'text': 'delivered'}
        if not stale.done():
            stale.cancel()
    asyncio.run(run())
