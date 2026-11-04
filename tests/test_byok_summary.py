"""Focused integration: BYOK routing and session ownership; expected 2s, max 15s."""
import httpx
import pytest
import main


@pytest.mark.asyncio
async def test_byok_summary_uses_selected_provider_and_never_persists_key(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'WEBUI_JOB_DIR', tmp_path / 'jobs')
    seen = {}
    async def run(config, system, prompt):
        seen.update(config=config, prompt=prompt)
        return '**Готово**'
    monkeypatch.setattr(main, '_run_byok', run)
    monkeypatch.setattr(main, '_omniroute_api_key', lambda: None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
        response = await client.post('/web/session', headers={'origin': 'http://testserver'})
        token = response.json()['token']
        sid = client.cookies.get(main.WEBUI_SESSION_COOKIE)
        transcript = main._store_webui_transcript(sid, 'Тестовый текст')
        payload = {'transcript_id': transcript, 'byok': {'apiFormat': 'responses', 'baseUrl': 'https://api.example.com/v1', 'modelId': 'sample', 'apiKey': 'secret-test-value'}}
        response = await client.post('/web/summarize', json=payload, headers={'origin': 'http://testserver', 'x-whisper-csrf': token})
        assert response.status_code == 200
        assert '<strong>Готово</strong>' in response.json()['summary_html']
        assert seen['config']['apiKey'] == 'secret-test-value' and seen['prompt'] == 'Тестовый текст'
        assert 'secret-test-value' not in response.text
        assert 'secret-test-value' not in repr(main._webui_transcripts)


@pytest.mark.asyncio
async def test_byok_rejects_private_endpoints():
    # Package-level adapter must reject local network targets before any request.
    with pytest.raises(main.HTTPException):
        await main._run_byok({'apiFormat': 'chat-completions', 'baseUrl': 'http://127.0.0.1:1', 'modelId': 'x', 'apiKey': 'test', 'maxOutputTokens': 10}, 'system', 'prompt')
