"""CPU-only API checks for cursor recovery and responsive upload cancellation."""
import asyncio
import json
import threading
import httpx
import pytest
from starlette.requests import Request
import main


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'WEBUI_JOB_DIR', tmp_path / 'jobs')
    monkeypatch.setattr(main.app.state, 'job_cancel_flags', {}, raising=False)
    main._webui_sessions.clear()


def request_for(owner, job_id):
    async def receive():
        return {'type': 'http.request', 'body': b'', 'more_body': False}
    return Request({'type': 'http', 'method': 'GET', 'path': f'/web/jobs/{job_id}/events',
        'headers': [(b'cookie', f'{main.WEBUI_SESSION_COOKIE}={owner}'.encode())]}, receive)


@pytest.mark.asyncio
async def test_delta_resume_reset_and_metadata_only_history():
    owner, _ = main._create_webui_session()
    store = main._job_store()
    job = store.create(owner, {'filename': 'delta.wav', 'size': 3, 'model': 'small', 'diarize': False})
    job_id = job['id']
    store.enqueue(owner, job_id, 1)
    store.start(owner, job_id, 1)
    segments = [{'start': i, 'end': i + 1, 'text': 'fragment ' + str(i)} for i in range(4)]
    for segment in segments:
        store.append_progress(owner, job_id, 1, segment=segment, progress={'phase': 'transcribing'})
    stream = await main.web_jobs_events(job_id, request_for(owner, job_id), cursor='1:0:2')
    event = json.loads((await anext(stream.body_iterator)).split('data: ', 1)[1])
    assert event['segments'] == segments[2:] and event['cursor'] == '1:0:4'
    assert not event['reset'] and 'segments' not in event['job']['result']
    await stream.body_iterator.aclose()
    store.append_progress(owner, job_id, 1, reset=True, progress={'phase': 'loading'})
    store.append_progress(owner, job_id, 1, segment=segments[0], progress={'phase': 'transcribing'})
    event = store.stream_delta(owner, job_id, '1:0:4')
    assert event['reset'] and event['cursor'] == '1:1:1' and event['segments'] == segments[:1]
    store.complete(owner, job_id, 1, {'text': segments[0]['text'], 'segments': segments[:1]})
    history = await main.web_jobs_list(request_for(owner, job_id))
    assert len(history['jobs']) == 1
    assert 'result' not in history['jobs'][0] and 'summaries' not in history['jobs'][0]
    assert store.get(owner, job_id)['result']['segments'] == segments[:1]


def test_bounded_replay_page_and_stale_cursor(tmp_path):
    store = main._job_store()
    job = store.create('owner', {})
    job_id = job['id']
    store.enqueue('owner', job_id, 1)
    store.start('owner', job_id, 1)
    segments = [{'start': i, 'end': i + 1, 'text': str(i)} for i in range(270)]
    store.complete('owner', job_id, 1, {'text': ' '.join(s['text'] for s in segments), 'segments': segments})
    first = store.stream_delta('owner', job_id, limit=128)
    second = store.stream_delta('owner', job_id, first['cursor'], limit=128)
    third = store.stream_delta('owner', job_id, second['cursor'], limit=128)
    assert [len(part['segments']) for part in [first, second, third]] == [128, 128, 14]
    assert first['has_more'] and second['has_more'] and not third['has_more']
    assert first['segments'] + second['segments'] + third['segments'] == segments
    assert store.stream_delta('owner', job_id, '0:0:0')['reset']


@pytest.mark.asyncio
@pytest.mark.parametrize(('disconnect', 'delete'), [(False, False), (True, False), (True, True)])
async def test_cancel_is_accepted_while_own_upload_fsync_waits(monkeypatch, disconnect, delete):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    spool = main._job_spool()
    original = spool.append
    def slow(*args):
        entered.set()
        assert release.wait(3)
        try:
            return original(*args)
        finally:
            finished.set()
    monkeypatch.setattr(spool, 'append', slow)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
        origin = {'origin': 'http://testserver'}
        token = (await client.post('/web/session', headers=origin)).json()['token']
        created = await client.post('/web/jobs', headers={**origin, 'x-whisper-csrf': token},
            json={'filename': 'slow.wav', 'size': 4, 'model': 'small', 'diarize': False})
        job_id = created.json()['id']
        token = created.headers['x-whisper-next-csrf']
        upload = asyncio.create_task(client.put(f'/web/jobs/{job_id}/upload?offset=0',
            content=b'abcd', headers={**origin, 'x-whisper-csrf': token}))
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            if disconnect:
                upload.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await upload
            control_token = (await client.post('/web/session', headers=origin)).json()['token']
            cancellation = await asyncio.wait_for(client.post(f'/web/jobs/{job_id}/cancel',
                headers={**origin, 'x-whisper-csrf': control_token}), .5)
            assert cancellation.status_code == 200 and cancellation.json()['status'] == 'cancelled'
            if delete:
                deleted = await client.delete(f'/web/jobs/{job_id}', headers={**origin,
                    'x-whisper-csrf': cancellation.headers['x-whisper-next-csrf']})
                assert deleted.status_code == 200
        finally:
            release.set()
            if not disconnect:
                response = await asyncio.wait_for(upload, 3)
            assert await asyncio.to_thread(finished.wait, 3)
            if getattr(main, '_upload_tasks', None):
                await asyncio.wait_for(asyncio.gather(*list(main._upload_tasks)), 3)
            if main._job_cleanup_tasks:
                await asyncio.wait_for(asyncio.gather(*list(main._job_cleanup_tasks)), 3)
        if not disconnect:
            assert response.status_code == 409
        assert spool.offset(job_id) == 0 and spool.usage()['reserved_bytes'] == 0

@pytest.mark.asyncio
async def test_sse_checks_owner_and_last_event_id_precedes_url_cursor():
    from fastapi import HTTPException
    owner, _ = main._create_webui_session()
    other, _ = main._create_webui_session()
    store = main._job_store()
    job_id = store.create(owner, {'filename': 'x', 'size': 1, 'model': 'small', 'diarize': False})['id']
    store.enqueue(owner, job_id, 1)
    store.start(owner, job_id, 1)
    segments = [{'text': str(index), 'start': index, 'end': index + 1} for index in range(3)]
    store.complete(owner, job_id, 1, {'text': '0 1 2', 'segments': segments})
    with pytest.raises(HTTPException) as forbidden:
        await main.web_jobs_events(job_id, request_for(other, job_id))
    assert forbidden.value.status_code == 404
    with pytest.raises(HTTPException) as invalid:
        await main.web_jobs_events(job_id, request_for(owner, job_id), cursor='bad')
    assert invalid.value.status_code == 400
    request = request_for(owner, job_id)
    request.scope['headers'].append((b'last-event-id', b'1:0:2'))
    stream = await main.web_jobs_events(job_id, request, cursor='1:0:0')
    first = await anext(stream.body_iterator)
    assert first.startswith('id: 1:0:3\n')
    assert json.loads(first.split('data: ', 1)[1])['segments'] == segments[2:]
    with pytest.raises(StopAsyncIteration):
        await anext(stream.body_iterator)
    assert job_id not in main._job_watchers


@pytest.mark.asyncio
@pytest.mark.parametrize('cancelled', [False, True])
async def test_worker_exit_is_deferred_or_cancelled_without_losing_audio(monkeypatch, cancelled):
    store, spool = main._job_store(), main._job_spool()
    settings = {'filename': 'retry.wav', 'size': 3, 'model': 'small', 'diarize': False}
    job_id = store.create('owner', settings)['id']
    spool.append(job_id, 0, b'abc', 3)
    store.enqueue('owner', job_id, 1)
    deferred = []
    monkeypatch.setattr(main, '_defer_or_launch_web_job', lambda owner, job: deferred.append((owner, job)), raising=False)
    async def lost_worker(*args, **kwargs):
        if cancelled:
            store.request_cancel('owner', job_id, 1)
        return main.JSONResponse({'error': 'worker ended', 'code': 'worker_stopped'}, status_code=503)
    monkeypatch.setattr(main, '_transcribe_impl', lost_worker)
    main.app.state.job_cancel_flags[(job_id, 1)] = cancelled
    await main._execute_web_job('owner', job_id, 1)
    result = store.get('owner', job_id)
    if cancelled:
        assert result['status'] == 'cancelled' and not deferred
        assert spool.offset(job_id) == 0
    else:
        assert result['status'] == 'queued' and result['attempt'] == 2
        assert result['result'] is None and result['error'] is None
        assert len(deferred) == 1 and deferred[0][1]['attempt'] == 2
        assert (spool.root / job_id).read_bytes() == b'abc'
    assert (job_id, 1) not in main.app.state.job_cancel_flags

