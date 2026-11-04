"""Focused integration: summary reuse and isolation, expected 3s, maximum 15s."""
import asyncio
import json
import httpx
import pytest
from diskcache import Cache
import main


@pytest.mark.asyncio
async def test_summary_cache_reuses_and_deduplicates_without_crossing_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'WEBUI_JOB_DIR', tmp_path / 'jobs')
    cache = Cache(str(tmp_path / 'cache'))
    monkeypatch.setattr(main, 'transcription_cache', cache)
    monkeypatch.setattr(main, '_omniroute_api_key', lambda: 'server-key')
    main._webui_summary_calls.clear()
    calls = []
    async def gateway(opener, request):
        calls.append('default')
        await asyncio.sleep(.05)
        return json.dumps({'model': 'test-model', 'choices': [{'message': {'content': '**Итог**'}}]})
    async def byok(config, system, prompt):
        calls.append(config['apiKey'])
        return 'Свой итог'
    monkeypatch.setattr(main, '_summary_open', gateway)
    monkeypatch.setattr(main, '_run_byok', byok)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
        first = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
        second = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
        sid = client.cookies.get(main.WEBUI_SESSION_COOKIE)
        one = main._store_webui_transcript(sid, 'Одинаковый текст')
        two = main._store_webui_transcript(sid, 'Одинаковый текст')
        async def send(token, body):
            return await client.post('/web/summarize', json=body, headers={'origin':'http://testserver','x-whisper-csrf':token})
        a, b = await asyncio.gather(send(first, {'transcript_id':one}), send(second, {'transcript_id':two}))
        assert a.status_code == b.status_code == 200
        assert sorted([a.json()['cached'], b.json()['cached']]) == [False, True]
        assert calls == ['default'] and b.json()['transcript_id'] == two
        token = b.headers['x-whisper-next-csrf']
        # Simulate reopening the persistent cache after a server restart.
        cache.close()
        cache = Cache(str(tmp_path / 'cache'))
        monkeypatch.setattr(main, 'transcription_cache', cache)
        reopened = await send(token, {'transcript_id': one})
        assert reopened.status_code == 200 and reopened.json()['cached']
        token = reopened.headers['x-whisper-next-csrf']
        other = {'apiFormat':'responses','baseUrl':'https://example.com/v1','modelId':'model-a','apiKey':'secret-key-a'}
        c = await send(token, {'transcript_id':one, 'byok':other})
        assert c.status_code == 200 and not c.json()['cached']
        token = c.headers['x-whisper-next-csrf']
        d = await send(token, {'transcript_id':one, 'byok':other})
        assert d.status_code == 200 and d.json()['cached']
        token = d.headers['x-whisper-next-csrf']
        e = await send(token, {'transcript_id':one, 'byok':{**other, 'apiKey':'secret-key-b'}})
        assert e.status_code == 200 and not e.json()['cached']
        token = e.headers['x-whisper-next-csrf']
        f = await send(token, {'transcript_id':one, 'force_refresh':True})
        assert f.status_code == 200 and not f.json()['cached']
        token = f.headers['x-whisper-next-csrf']
        g = await send(token, {'transcript_id':one, 'instruction':'Другой формат'})
        assert g.status_code == 200 and not g.json()['cached']
        assert calls == ['default', 'secret-key-a', 'secret-key-b', 'default', 'default']
        assert all('secret-key' not in key for key in cache.iterkeys())
    cache.close()
