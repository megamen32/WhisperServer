"""Focused, CPU-only checks for the web job integration prerequisites."""
import asyncio
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

import main


@pytest.fixture(autouse=True)
def isolated_webui(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'WEBUI_JOB_DIR', tmp_path / 'jobs')
    # ASGITransport intentionally skips lifespan; tests inject their own channel.
    monkeypatch.setattr(main.app.state, 'job_cancel_flags', {}, raising=False)
    monkeypatch.setattr(main.app.state, 'worker_accepting', True, raising=False)
    monkeypatch.setattr(main.app.state, 'model_process', type('Alive', (), {'is_alive': lambda self: True})(), raising=False)
    main._webui_sessions.clear()
    main._webui_transcripts.clear()


@pytest.mark.asyncio
async def test_job_upload_worker_reload_and_summary(monkeypatch):
    # Focused integration: a detached worker result remains accessible after a
    # reload and can be summarized; expected <1s, maximum 5s, no GPU.
    entered, release = asyncio.Event(), asyncio.Event()
    async def transcribe(file, model, language, beam, temperature, stream, words, vad, diarize, **kwargs):
        assert await file.read() == b'abcdef'
        assert diarize is True
        entered.set()
        await release.wait()
        return {'text': 'hello', 'segments': []}
    monkeypatch.setattr(main, '_transcribe_impl', transcribe)
    monkeypatch.setattr(main, '_omniroute_api_key', lambda: 'test')
    async def summary(*args):
        return '{"choices":[{"message":{"content":"**short**"}}]}'
    monkeypatch.setattr(main, '_summary_open', summary)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
        token = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
        async def mutate(method, path, **kwargs):
            nonlocal token
            response = await client.request(method, path, headers={'origin': 'http://testserver', 'x-whisper-csrf': token}, **kwargs)
            token = response.headers.get('x-whisper-next-csrf', token)
            return response
        job = (await mutate('POST', '/web/jobs', json={'filename': 'a.wav', 'size': 6, 'model': 'small', 'diarize': True})).json()
        job_id = job['id']
        assert (await mutate('PUT', f'/web/jobs/{job_id}/upload?offset=0', content=b'abc')).json()['offset'] == 3
        assert (await mutate('PUT', f'/web/jobs/{job_id}/upload?offset=0', content=b'abc')).json()['offset'] == 3
        assert (await mutate('POST', f'/web/jobs/{job_id}/start')).status_code == 409
        assert (await mutate('PUT', f'/web/jobs/{job_id}/upload?offset=3', content=b'def')).json()['offset'] == 6
        assert (await mutate('POST', f'/web/jobs/{job_id}/start')).json()['status'] == 'queued'
        await asyncio.wait_for(entered.wait(), 2)
        assert (await client.get(f'/web/jobs/{job_id}')).json()['status'] == 'processing'
        release.set()
        await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 2)
        main._webui_sessions.clear()
        token = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
        restored = (await client.get(f'/web/jobs/{job_id}')).json()
        assert restored['status'] == 'done' and restored['result']['text'] == 'hello'
        assert (await client.get('/web/jobs')).json()['jobs'][0]['id'] == job_id
        assert (await client.get('/')).status_code == 200  # another tab rotates its displayed token
        response = await mutate('POST', '/web/summarize', json={'transcript_id': job_id})
        assert response.status_code == 200 and response.json()['summary'] == '**short**'
        assert response.json()['summary_html'] == '<p><strong>short</strong></p>\n'
        saved = (await client.get(f'/web/jobs/{job_id}')).json()['summaries']['latest']
        assert saved['summary'] == '**short**'
        assert saved['summary_html'] == response.json()['summary_html']


@pytest.mark.asyncio
async def test_real_worker_queue_cancel_discards_late_result(monkeypatch):
    # Focused integration: exercise actual queue/listener and cancellation race;
    # defect: a late result overwrites cancellation. Expected <1s, maximum 5s.
    requests, responses = queue.Queue(), queue.Queue()
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(main.app.state, 'request_queue', requests, raising=False)
    monkeypatch.setattr(main.app.state, 'executor', executor, raising=False)
    flags = {}
    monkeypatch.setattr(main.app.state, 'job_cancel_flags', flags, raising=False)
    monkeypatch.setattr(main.app.state, 'model_process', type('Alive', (), {'is_alive': lambda self: True})(), raising=False)
    monkeypatch.setattr(main, '_get_cached_result', lambda key: None)
    monkeypatch.setenv('WHISPER_DIARIZATION_TIMEOUT_SECONDS', '0')
    listener = asyncio.create_task(main.response_listener(None, responses))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
            token = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
            async def mutate(method, path, **kwargs):
                nonlocal token
                response = await client.request(method, path, headers={'origin': 'http://testserver', 'x-whisper-csrf': token}, **kwargs)
                token = response.headers.get('x-whisper-next-csrf', token)
                return response
            job = (await mutate('POST', '/web/jobs', json={'filename': 'x', 'size': 3, 'model': 'small', 'diarize': True})).json()
            job_id = job['id']
            await mutate('PUT', f'/web/jobs/{job_id}/upload?offset=0', content=b'abc')
            await mutate('POST', f'/web/jobs/{job_id}/start')
            for _ in range(100):
                if not requests.empty():
                    break
                await asyncio.sleep(.01)
            worker_request = requests.get_nowait()
            assert worker_request['audio_bytes'] == b'abc'
            assert (await mutate('POST', f'/web/jobs/{job_id}/cancel')).json()['status'] == 'cancel_requested'
            assert flags[(job_id, 1)] is True
            assert worker_request['web_job_key'] == (job_id, 1)
            await asyncio.sleep(.05)
            assert (await client.get(f'/web/jobs/{job_id}')).json()['status'] == 'cancel_requested'
            assert main._job_spool().offset(job_id) == 3
            responses.put({'request_id': worker_request['request_id'], 'result': {'text': 'late'}, 'final': True})
            await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 2)
            job = (await client.get(f'/web/jobs/{job_id}')).json()
            assert job['status'] == 'cancelled' and job['result'] is None
            assert main._job_spool().offset(job_id) == 0
            assert flags == {}
    finally:
        responses.put(None)
        await asyncio.wait_for(listener, 2)
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_diarization_change_retries_after_current_attempt(monkeypatch):
    # Focused integration: changing a running attempt schedules exactly one
    # replacement with the desired setting. Expected <1s, maximum 5s.
    if not hasattr(main.JobStore, 'retry_if_diarization_changed') or not hasattr(main.JobStore, 'set_diarization'):
        pytest.skip('JobStore deferred retry contract is not installed')
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = []
    async def transcribe(file, model, language, beam, temperature, stream, words, vad, diarize, **kwargs):
        calls.append(diarize)
        if len(calls) == 1:
            entered.set()
            await release.wait()
        else:
            finished.set()
        return {'text': str(diarize)}
    monkeypatch.setattr(main, '_transcribe_impl', transcribe)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
        token = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
        async def mutate(method, path, **kwargs):
            nonlocal token
            response = await client.request(method, path, headers={'origin': 'http://testserver', 'x-whisper-csrf': token}, **kwargs)
            token = response.headers.get('x-whisper-next-csrf', token)
            return response
        job = (await mutate('POST', '/web/jobs', json={'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False})).json()
        job_id = job['id']
        await mutate('PUT', f'/web/jobs/{job_id}/upload?offset=0', content=b'abc')
        await mutate('POST', f'/web/jobs/{job_id}/start')
        await asyncio.wait_for(entered.wait(), 2)
        response = await mutate('PATCH', f'/web/jobs/{job_id}', json={'diarize': True})
        assert response.status_code == 200 and response.json()['diarize'] is True
        release.set()
        await asyncio.wait_for(finished.wait(), 2)
        await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 2)
        job = (await client.get(f'/web/jobs/{job_id}')).json()
        assert calls == [False, True]
        assert job['status'] == 'done' and job['attempt'] == 2 and job['result']['text'] == 'True'


@pytest.mark.asyncio
async def test_restart_recovers_inflight_desired_diarization(monkeypatch):
    # Focused integration: restart requeues interrupted work with the latest
    # requested setting and the persisted audio. Expected <1s, maximum 5s.
    store = main._job_store()
    settings = {'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False, 'desired_diarize': False}
    job = store.create('owner', settings)
    main._job_spool().append(job['id'], 0, b'abc', 3)
    store.enqueue('owner', job['id'], 1)
    store.start('owner', job['id'], 1)
    store.set_diarization('owner', job['id'], 1, {**settings, 'desired_diarize': True})
    calls = []
    async def transcribe(file, model, language, beam, temperature, stream, words, vad, diarize, **kwargs):
        calls.append((await file.read(), diarize))
        return {'text': 'recovered'}
    monkeypatch.setattr(main, '_transcribe_impl', transcribe)
    recovered = main._recover_web_jobs(main._job_store())
    assert len(recovered) == 1 and recovered[0]['owner'] == 'owner'
    for item in recovered:
        main._launch_job(item['owner'], item)
    await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 2)
    job = main._job_store().get('owner', job['id'])
    assert calls == [(b'abc', True)]
    assert job['status'] == 'done' and job['attempt'] == 2 and job['result']['text'] == 'recovered'


def test_restart_recovers_failed_attempt_with_pending_setting():
    # Fast unit: a crash between fail and deferred retry must not lose the
    # requested setting. Expected <1s, maximum 5s.
    store = main._job_store()
    settings = {'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False, 'desired_diarize': False}
    job = store.create('owner', settings)
    store.enqueue('owner', job['id'], 1)
    store.start('owner', job['id'], 1)
    store.set_diarization('owner', job['id'], 1, {**settings, 'desired_diarize': True})
    store.fail('owner', job['id'], 1, 'worker failed')
    recovered = main._recover_web_jobs(main._job_store())
    assert len(recovered) == 1
    assert recovered[0]['id'] == job['id'] and recovered[0]['attempt'] == 2
    assert recovered[0]['status'] == 'queued' and recovered[0]['diarization']['diarize'] is True


@pytest.mark.asyncio
async def test_tabs_lost_response_and_repeated_start(monkeypatch):
    # Focused integration: another tab and a lost response cannot invalidate
    # the first tab or enqueue the same upload twice. Expected <1s, max 5s.
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def transcribe(*args, **kwargs):
        calls.append(1)
        entered.set()
        await release.wait()
        return {'text': 'ok'}
    monkeypatch.setattr(main, '_transcribe_impl', transcribe)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
        first = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
        headers = {'origin': 'http://testserver', 'x-whisper-csrf': first}
        second = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
        created = await client.post('/web/jobs', headers=headers, json={'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False})
        assert created.status_code == 200
        job_id = created.json()['id']
        assert (await client.patch(f'/web/jobs/{job_id}', headers=headers, json={'diarize': True})).status_code == 403
        headers['x-whisper-csrf'] = second
        patched = await client.patch(f'/web/jobs/{job_id}', headers=headers, json={'diarize': True})
        assert patched.status_code == 200
        headers['x-whisper-csrf'] = patched.headers['x-whisper-next-csrf']
        uploaded = await client.put(f'/web/jobs/{job_id}/upload?offset=0', headers=headers, content=b'abc')
        assert uploaded.status_code == 200
        headers['x-whisper-csrf'] = uploaded.headers['x-whisper-next-csrf']
        assert (await client.post(f'/web/jobs/{job_id}/start', headers=headers)).status_code == 200
        await asyncio.wait_for(entered.wait(), 2)
        assert (await client.post(f'/web/jobs/{job_id}/start', headers=headers)).status_code == 403
        headers['x-whisper-csrf'] = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
        repeated = await client.post(f'/web/jobs/{job_id}/start', headers=headers)
        assert repeated.status_code == 200 and repeated.json()['status'] == 'processing'
        assert calls == [1]
        headers['x-whisper-csrf'] = repeated.headers['x-whisper-next-csrf']
        assert (await client.post(f'/web/jobs/{job_id}/cancel', headers=headers)).json()['status'] == 'cancel_requested'
        release.set()
        await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 2)
        assert (await client.get(f'/web/jobs/{job_id}')).json()['status'] == 'cancelled'


@pytest.mark.asyncio
async def test_summary_error_returns_rotated_csrf():
    # Purpose: recover a token after a failed summary; defect: token lost on 404.
    # Category: focused integration; expected <1s, maximum 5s.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://testserver") as client:
        session = await client.post('/web/session', headers={'origin': 'http://testserver'})
        token = session.json()['token']
        response = await client.post('/web/summarize', json={'transcript_id': 'missing'},
                                     headers={'origin': 'http://testserver', 'x-whisper-csrf': token})
        assert response.status_code == 404
        replacement = response.headers['x-whisper-next-csrf']
        assert replacement != token
        response = await client.post('/web/summarize', json={'transcript_id': 'missing'},
                                     headers={'origin': 'http://testserver', 'x-whisper-csrf': replacement})
        assert response.status_code == 404
        assert 'x-whisper-next-csrf' in response.headers


@pytest.mark.asyncio
async def test_summary_gateway_does_not_block_event_loop():
    # Purpose: permit polling during gateway I/O; defect: synchronous open blocked loop.
    # Category: fast unit; expected <1s, maximum 5s.
    started, release = threading.Event(), threading.Event()

    class Response:
        def __enter__(self):
            started.set()
            assert release.wait(3)
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b'ok'

    class Opener:
        def open(self, *args, **kwargs):
            return Response()

    task = asyncio.create_task(main._summary_open(Opener(), object()))
    try:
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set()
        assert not task.done()
    finally:
        release.set()
    assert await asyncio.wait_for(task, 3) == 'ok'


@pytest.mark.asyncio
async def test_summary_rejects_changed_attempt(monkeypatch):
    # Focused integration: stale summary must neither be saved nor returned;
    # expected <1s, maximum 5s.
    entered, release = asyncio.Event(), asyncio.Event()
    async def summary(*args):
        entered.set()
        await release.wait()
        return '{"choices":[{"message":{"content":"old summary"}}]}'
    monkeypatch.setattr(main, '_summary_open', summary)
    monkeypatch.setattr(main, '_omniroute_api_key', lambda: 'test')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
        token = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
        owner = client.cookies[main.WEBUI_SESSION_COOKIE]
        store = main._job_store()
        settings = {'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False, 'desired_diarize': False}
        job = store.create(owner, settings)
        store.enqueue(owner, job['id'], 1)
        store.start(owner, job['id'], 1)
        store.complete(owner, job['id'], 1, {'text': 'old transcript'})
        task = asyncio.create_task(client.post('/web/summarize', json={'transcript_id': job['id']},
                                               headers={'origin': 'http://testserver', 'x-whisper-csrf': token}))
        await asyncio.wait_for(entered.wait(), 2)
        store.retry(owner, job['id'], settings)
        store.start(owner, job['id'], 2)
        store.complete(owner, job['id'], 2, {'text': 'new transcript'})
        release.set()
        response = await asyncio.wait_for(task, 2)
        assert response.status_code == 409
        assert 'old summary' not in response.text
        assert store.get(owner, job['id'])['summaries'] == {}


@pytest.mark.asyncio
async def test_delete_and_cancel_free_full_spool(monkeypatch):
    # Focused integration: terminal deletion and pre-start cancellation reclaim
    # full spool quota, preserving completed audio until deletion. Max 5s.
    monkeypatch.setattr(main, 'WEBUI_MAX_SPOOL_BYTES', 3)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
        token = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
        async def mutate(method, path, **kwargs):
            nonlocal token
            response = await client.request(method, path, headers={'origin': 'http://testserver', 'x-whisper-csrf': token}, **kwargs)
            token = response.headers.get('x-whisper-next-csrf', token)
            return response
        settings = {'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False}
        first = (await mutate('POST', '/web/jobs', json=settings)).json()['id']
        second = (await mutate('POST', '/web/jobs', json=settings)).json()['id']
        assert (await mutate('PUT', f'/web/jobs/{first}/upload?offset=0', content=b'abc')).status_code == 200
        assert (await mutate('PUT', f'/web/jobs/{second}/upload?offset=0', content=b'def')).status_code == 400
        assert (await mutate('DELETE', f'/web/jobs/{first}')).status_code == 409
        owner = client.cookies[main.WEBUI_SESSION_COOKIE]
        store = main._job_store()
        store.enqueue(owner, first, 1)
        store.start(owner, first, 1)
        store.complete(owner, first, 1, {'text': 'done'})
        assert main._job_spool().offset(first) == 3
        assert (await mutate('DELETE', f'/web/jobs/{first}')).status_code == 200
        assert (await client.get(f'/web/jobs/{first}')).status_code == 404
        assert (await mutate('PUT', f'/web/jobs/{second}/upload?offset=0', content=b'def')).status_code == 200
        assert (await mutate('POST', f'/web/jobs/{second}/cancel')).json()['status'] == 'cancelled'
        assert main._job_spool().offset(second) == 0
        third = (await mutate('POST', '/web/jobs', json=settings)).json()['id']
        assert (await mutate('PUT', f'/web/jobs/{third}/upload?offset=0', content=b'ghi')).status_code == 200


@pytest.mark.asyncio
async def test_timeout_response_does_not_confirm_cancellation(monkeypatch):
    # Focused integration: an unexpected timeout is not worker completion;
    # cancelled data must remain until a terminal signal. Max 5s.
    entered, release = asyncio.Event(), asyncio.Event()
    async def timeout(*args, **kwargs):
        entered.set()
        await release.wait()
        return main.JSONResponse({'error': 'Timeout'}, status_code=504)
    monkeypatch.setattr(main, '_transcribe_impl', timeout)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
        token = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
        async def mutate(method, path, **kwargs):
            nonlocal token
            response = await client.request(method, path, headers={'origin': 'http://testserver', 'x-whisper-csrf': token}, **kwargs)
            token = response.headers.get('x-whisper-next-csrf', token)
            return response
        job_id = (await mutate('POST', '/web/jobs', json={'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False})).json()['id']
        await mutate('PUT', f'/web/jobs/{job_id}/upload?offset=0', content=b'abc')
        await mutate('POST', f'/web/jobs/{job_id}/start')
        await asyncio.wait_for(entered.wait(), 2)
        await mutate('POST', f'/web/jobs/{job_id}/cancel')
        release.set()
        await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 2)
        assert (await client.get(f'/web/jobs/{job_id}')).json()['status'] == 'cancel_requested'
        assert main._job_spool().offset(job_id) == 3
        assert (await mutate('DELETE', f'/web/jobs/{job_id}')).status_code == 409


@pytest.mark.asyncio
async def test_idempotent_create_http_retry_and_owner_isolation():
    # Focused integration: lost create response, reload and simultaneous tabs;
    # defect: duplicate or lost job. Expected <1s, maximum 5s.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
        origin = {'origin': 'http://testserver'}
        token1 = (await client.post('/web/session', headers=origin)).json()['token']
        token2 = (await client.post('/web/session', headers=origin)).json()['token']
        body = {'filename': 'x.wav', 'size': 3, 'model': 'small', 'diarize': False, 'client_request_id': 'request-123'}
        responses = await asyncio.gather(*(client.post('/web/jobs', json=body, headers={**origin, 'x-whisper-csrf': token}) for token in (token1, token2)))
        assert [r.status_code for r in responses] == [200, 200]
        job_id = responses[0].json()['id']
        assert responses[1].json()['id'] == job_id
        main._webui_sessions.clear()
        token = (await client.post('/web/session', headers=origin)).json()['token']
        replay = await client.post('/web/jobs', json=body, headers={**origin, 'x-whisper-csrf': token})
        assert replay.status_code == 200 and replay.json()['id'] == job_id
        token = replay.headers['x-whisper-next-csrf']
        changed = await client.post('/web/jobs', json={**body, 'size': 4}, headers={**origin, 'x-whisper-csrf': token})
        assert changed.status_code == 409
        assert len((await client.get('/web/jobs')).json()['jobs']) == 1
        token = changed.headers['x-whisper-next-csrf']
        cancelled = await client.post(f'/web/jobs/{job_id}/cancel', headers={**origin, 'x-whisper-csrf': token})
        token = cancelled.headers['x-whisper-next-csrf']
        deleted = await client.delete(f'/web/jobs/{job_id}', headers={**origin, 'x-whisper-csrf': token})
        assert deleted.status_code == 200
        token = deleted.headers['x-whisper-next-csrf']
        gone = await client.post('/web/jobs', json=body, headers={**origin, 'x-whisper-csrf': token})
        assert gone.status_code == 410
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as other:
        token = (await other.post('/web/session', headers=origin)).json()['token']
        response = await other.post('/web/jobs', json=body, headers={**origin, 'x-whisper-csrf': token})
        assert response.status_code == 200 and response.json()['id'] != job_id


def test_idempotent_create_concurrent_transactions():
    # Focused integration: SQLite serializes separate writers and commits the
    # request key with the job. Expected <1s, maximum 5s.
    barrier = threading.Barrier(2)
    settings = {'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False, 'desired_diarize': False}
    def create():
        store = main._job_store()
        barrier.wait(timeout=3)
        return main._create_idempotent_job(store, 'owner', 'same-request', settings)['id']
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(create) for _ in range(2)]
        ids = [future.result(timeout=5) for future in futures]
    assert ids[0] == ids[1]
    assert len(main._job_store().list('owner')) == 1


@pytest.mark.parametrize('stage', ['queued', 'diarization', 'segments'])
def test_worker_cooperative_cancellation(stage, monkeypatch):
    # Focused integration: real worker loop, CPU-only fake model; cancellation
    # stops queued work, after diarization returns, or between segments.
    # Expected <1s, maximum 5s.
    from types import SimpleNamespace
    requests, responses = queue.Queue(), queue.Queue()
    stop = threading.Event()
    entered, release = threading.Event(), threading.Event()
    flags = {('job', 1): stage == 'queued'}
    actions = []
    monkeypatch.setattr(main, 'audio_duration_seconds', lambda path: 1)
    monkeypatch.setattr(main, 'reset_cuda_peak_memory', lambda: None)
    monkeypatch.setattr(main, 'peak_vram_mb', lambda: 0)
    monkeypatch.setattr(main, 'append_metric', lambda metric: None)
    monkeypatch.setattr(main.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(main, 'ensure_vram_for_model', lambda *args: None)
    def diarize(*args, **kwargs):
        actions.append('diarize')
        entered.set()
        assert release.wait(3)
        return []
    monkeypatch.setattr(main, 'diarize_audio', diarize)
    class Model:
        def transcribe(self, *args, **kwargs):
            actions.append('transcribe')
            def segments():
                yield SimpleNamespace(start=0, end=1, text='first', words=[])
                entered.set()
                assert release.wait(3)
                yield SimpleNamespace(start=1, end=2, text='second', words=[])
                actions.append('third')
            return segments(), SimpleNamespace(language='en', language_probability=1)
    monkeypatch.setattr(main, '_create_model_entry', lambda name: Model())
    requests.put({'request_id': 'r', 'audio_bytes': b'a', 'model': 'small', 'stream': False,
                  'diarize': stage == 'diarization', 'web_job_key': ('job', 1)})
    worker = threading.Thread(target=main.model_worker, args=(requests, responses, stop, {}, flags))
    worker.start()
    try:
        if stage != 'queued':
            assert entered.wait(3)
            flags[('job', 1)] = True
            assert responses.empty()  # no acknowledgement while blocked in an opaque call
            release.set()
        response = responses.get(timeout=3)
        assert response['error_code'] == 'web_job_cancelled'
        assert 'result' not in response
        if stage == 'queued':
            assert not actions
        elif stage == 'diarization':
            assert actions == ['diarize']
        else:
            assert actions == ['transcribe']
    finally:
        release.set()
        stop.set()
        worker.join(timeout=3)
        assert not worker.is_alive()


@pytest.mark.asyncio
async def test_http_cancel_stops_worker_between_segments(monkeypatch):
    # Focused integration: HTTP -> shared flag -> worker -> terminal status,
    # without GPU. Expected <1s, maximum 5s.
    from types import SimpleNamespace
    requests, responses = queue.Queue(), queue.Queue()
    stop, entered, release = threading.Event(), threading.Event(), threading.Event()
    flags, seen = {}, []
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(main.app.state, 'request_queue', requests, raising=False)
    monkeypatch.setattr(main.app.state, 'executor', executor, raising=False)
    monkeypatch.setattr(main.app.state, 'job_cancel_flags', flags, raising=False)
    monkeypatch.setattr(main.app.state, 'model_process', type('Alive', (), {'is_alive': lambda self: True})(), raising=False)
    monkeypatch.setattr(main, '_get_cached_result', lambda key: None)
    monkeypatch.setattr(main, 'audio_duration_seconds', lambda path: 1)
    monkeypatch.setattr(main, 'reset_cuda_peak_memory', lambda: None)
    monkeypatch.setattr(main, 'peak_vram_mb', lambda: 0)
    monkeypatch.setattr(main, 'append_metric', lambda metric: None)
    monkeypatch.setattr(main.torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(main, 'ensure_vram_for_model', lambda *args: None)
    class Model:
        def transcribe(self, *args, **kwargs):
            def segments():
                yield SimpleNamespace(start=0, end=1, text='first', words=[])
                entered.set()
                assert release.wait(3)
                yield SimpleNamespace(start=1, end=2, text='second', words=[])
                seen.append('continued')
            return segments(), SimpleNamespace(language='en', language_probability=1)
    monkeypatch.setattr(main, '_create_model_entry', lambda name: Model())
    worker = threading.Thread(target=main.model_worker, args=(requests, responses, stop, {}, flags))
    worker.start()
    listener = asyncio.create_task(main.response_listener(None, responses))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
            token = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
            async def mutate(method, path, **kwargs):
                nonlocal token
                response = await client.request(method, path, headers={'origin': 'http://testserver', 'x-whisper-csrf': token}, **kwargs)
                token = response.headers.get('x-whisper-next-csrf', token)
                return response
            job_id = (await mutate('POST', '/web/jobs', json={'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False})).json()['id']
            await mutate('PUT', f'/web/jobs/{job_id}/upload?offset=0', content=b'abc')
            await mutate('POST', f'/web/jobs/{job_id}/start')
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(.01)
            assert entered.is_set()
            assert (await mutate('POST', f'/web/jobs/{job_id}/cancel')).json()['status'] == 'cancel_requested'
            release.set()
            await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 3)
            assert (await client.get(f'/web/jobs/{job_id}')).json()['status'] == 'cancelled'
            assert seen == [] and main._job_spool().offset(job_id) == 0 and flags == {}
    finally:
        release.set()
        stop.set()
        worker.join(timeout=3)
        responses.put(None)
        await asyncio.wait_for(listener, 3)
        executor.shutdown(wait=True)
        assert not worker.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_first', [False, True])
async def test_worker_death_unblocks_job_and_next_job(monkeypatch, cancel_first):
    # Focused integration: worker death resolves the active job and releases
    # the shared slot for the next one. Expected <1s, maximum 5s; no GPU.
    class Process:
        alive = True
        def is_alive(self):
            return self.alive
    process = Process()
    requests = queue.Queue()
    monkeypatch.setattr(main.app.state, 'model_process', process, raising=False)
    monkeypatch.setattr(main.app.state, 'request_queue', requests, raising=False)
    monkeypatch.setattr(main, '_get_cached_result', lambda key: None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
        token = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
        async def mutate(method, path, **kwargs):
            nonlocal token
            response = await client.request(method, path, headers={'origin': 'http://testserver', 'x-whisper-csrf': token}, **kwargs)
            token = response.headers.get('x-whisper-next-csrf', token)
            return response
        async def start_job():
            job_id = (await mutate('POST', '/web/jobs', json={'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False})).json()['id']
            await mutate('PUT', f'/web/jobs/{job_id}/upload?offset=0', content=b'abc')
            started = await mutate('POST', f'/web/jobs/{job_id}/start')
            return job_id, started
        first, started = await start_job()
        assert started.status_code == 200
        for _ in range(100):
            if not requests.empty():
                break
            await asyncio.sleep(.01)
        assert not requests.empty()
        requests.get_nowait()
        if cancel_first:
            assert (await mutate('POST', f'/web/jobs/{first}/cancel')).json()['status'] == 'cancel_requested'
        process.alive = False
        await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 3)
        job = (await client.get(f'/web/jobs/{first}')).json()
        assert job['status'] == ('cancelled' if cancel_first else 'error')
        if not cancel_first:
            assert 'worker stopped' in job['error'].lower()
        second, started = await start_job()
        assert started.status_code == 503
        await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 3)
        assert (await client.get(f'/web/jobs/{second}')).json()['status'] == 'uploading'
        assert requests.empty()  # a dead worker must not accumulate queued audio
        assert not main.pending_results


@pytest.mark.asyncio
async def test_worker_death_drains_already_published_result(monkeypatch):
    # Focused integration: listener may dispatch the final worker reply after
    # process death is observed. Expected <2s, maximum 5s; no GPU.
    class Process:
        alive = True
        def is_alive(self):
            return self.alive
    process = Process()
    requests, responses = queue.Queue(), queue.Queue()
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(main.app.state, 'model_process', process, raising=False)
    monkeypatch.setattr(main.app.state, 'request_queue', requests, raising=False)
    monkeypatch.setattr(main.app.state, 'executor', executor, raising=False)
    monkeypatch.setattr(main, '_get_cached_result', lambda key: None)
    listener = None
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
            token = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
            async def mutate(method, path, **kwargs):
                nonlocal token
                response = await client.request(method, path, headers={'origin': 'http://testserver', 'x-whisper-csrf': token}, **kwargs)
                token = response.headers.get('x-whisper-next-csrf', token)
                return response
            job_id = (await mutate('POST', '/web/jobs', json={'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False})).json()['id']
            await mutate('PUT', f'/web/jobs/{job_id}/upload?offset=0', content=b'abc')
            await mutate('POST', f'/web/jobs/{job_id}/start')
            for _ in range(100):
                if not requests.empty():
                    break
                await asyncio.sleep(.01)
            request = requests.get_nowait()
            responses.put({'request_id': request['request_id'], 'result': {'text': 'finished'}, 'final': True})
            process.alive = False
            await asyncio.sleep(.3)
            listener = asyncio.create_task(main.response_listener(None, responses))
            await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 3)
            job = (await client.get(f'/web/jobs/{job_id}')).json()
            assert job['status'] == 'done' and job['result']['text'] == 'finished'
    finally:
        if listener is None:
            listener = asyncio.create_task(main.response_listener(None, responses))
        responses.put(None)
        await asyncio.wait_for(listener, 3)
        executor.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel_first', [False, True])
async def test_worker_death_recovers_cached_result(monkeypatch, tmp_path, cancel_first):
    # Focused integration: worker cached its result but died before delivering
    # a reply; cancellation must still win. Expected <2s, maximum 5s, no GPU.
    class Process:
        alive = True
        def is_alive(self):
            return self.alive
    process, requests = Process(), queue.Queue()
    cache = main.Cache(str(tmp_path / 'transcription-cache'))
    monkeypatch.setattr(main, 'transcription_cache', cache)
    monkeypatch.setattr(main.app.state, 'model_process', process, raising=False)
    monkeypatch.setattr(main.app.state, 'request_queue', requests, raising=False)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
            token = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
            async def mutate(method, path, **kwargs):
                nonlocal token
                response = await client.request(method, path, headers={'origin': 'http://testserver', 'x-whisper-csrf': token}, **kwargs)
                token = response.headers.get('x-whisper-next-csrf', token)
                return response
            job_id = (await mutate('POST', '/web/jobs', json={'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False})).json()['id']
            await mutate('PUT', f'/web/jobs/{job_id}/upload?offset=0', content=b'abc')
            await mutate('POST', f'/web/jobs/{job_id}/start')
            for _ in range(100):
                if not requests.empty():
                    break
                await asyncio.sleep(.01)
            request = requests.get_nowait()
            cache.set(request['cache_key'], {'text': 'cached result', 'client_comment': 'another client'})
            if cancel_first:
                assert (await mutate('POST', f'/web/jobs/{job_id}/cancel')).json()['status'] == 'cancel_requested'
            process.alive = False  # No response is ever delivered.
            await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 3)
            job = (await client.get(f'/web/jobs/{job_id}')).json()
            if cancel_first:
                assert job['status'] == 'cancelled' and job['result'] is None
            else:
                assert job['status'] == 'done' and job['result']['text'] == 'cached result'
                assert 'client_comment' not in job['result']
            assert not main.pending_results
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_shutdown_clears_manager_proxy_after_job_cleanup(monkeypatch):
    # Fast unit: shutdown must not expose a dead Manager proxy to a later
    # lifespan and must let job cleanup finish while the proxy lives. Max 5s.
    finished = asyncio.Event()
    flags = {('job', 1): True}
    async def job():
        try:
            await asyncio.Event().wait()
        finally:
            flags.pop(('job', 1))
            finished.set()
    task = asyncio.create_task(job())
    await asyncio.sleep(0)
    monkeypatch.setattr(main, '_job_tasks', {task})
    class Manager:
        called = False
        def shutdown(self):
            assert finished.is_set() and not flags
            assert main.app.state.job_cancel_flags is None
            self.called = True
    class Process:
        def join(self, timeout): pass
        def is_alive(self): return False
    class Executor:
        def shutdown(self, wait): pass
    class FullRequestQueue:
        cancelled = False
        closed = False
        def __init__(self):
            self.queue = main.mp.get_context('spawn').Queue()
            self.queue.put(b'x' * (2 * 1024 * 1024))  # no reader; feeder fills its pipe
        def cancel_join_thread(self):
            self.cancelled = True
            self.queue.cancel_join_thread()
        def close(self):
            self.closed = True
            self.queue.close()
        def join_thread(self): raise AssertionError('shutdown must not join a full pipe')
    async def no_bot(): pass
    async def monitor(): await asyncio.Event().wait()
    monkeypatch.setattr(main, 'stop_bot_subprocess', no_bot)
    monkeypatch.setattr(main.app.state, 'job_cancel_flags', flags, raising=False)
    manager = Manager()
    monkeypatch.setattr(main.app.state, 'manager', manager, raising=False)
    monkeypatch.setattr(main.app.state, 'model_process', Process(), raising=False)
    monkeypatch.setattr(main.app.state, 'stop_event', threading.Event(), raising=False)
    monkeypatch.setattr(main.app.state, 'lifecycle_stop', asyncio.Event(), raising=False)
    monkeypatch.setattr(main.app.state, 'bot_monitor_task', asyncio.create_task(monitor()), raising=False)
    monkeypatch.setattr(main.app.state, 'response_listener_retire', None, raising=False)
    monkeypatch.setattr(main.app.state, 'response_listener_task', asyncio.create_task(monitor()), raising=False)
    full_queue = FullRequestQueue()
    monkeypatch.setattr(main.app.state, 'request_queue', full_queue, raising=False)
    monkeypatch.setattr(main.app.state, 'response_queue', queue.Queue(), raising=False)
    monkeypatch.setattr(main.app.state, 'executor', Executor(), raising=False)
    await asyncio.wait_for(main.shutdown_event(), 2)
    assert finished.is_set() and main.app.state.job_cancel_flags is None and manager.called
    assert full_queue.cancelled and full_queue.closed
    assert not hasattr(main.app.state, 'worker_accepting')
    main.app.state.worker_accepting = False  # allow the fixture to restore prior state
    main.app.state.manager = manager  # allow monkeypatch to restore prior state


@pytest.mark.asyncio
async def test_active_cancellation_requires_channel(monkeypatch):
    # Focused integration: an unavailable channel is reported explicitly and
    # does not falsely claim that cancellation was sent. Max 5s.
    monkeypatch.setattr(main.app.state, 'job_cancel_flags', None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
        token = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
        owner = client.cookies[main.WEBUI_SESSION_COOKIE]
        store = main._job_store()
        job = store.create(owner, {'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False})
        store.enqueue(owner, job['id'], 1)
        store.start(owner, job['id'], 1)
        response = await client.post(f"/web/jobs/{job['id']}/cancel", headers={'origin': 'http://testserver', 'x-whisper-csrf': token})
        assert response.status_code == 503
        assert store.get(owner, job['id'])['status'] == 'processing'


@pytest.mark.asyncio
async def test_supervisor_replaces_dead_worker_without_replaying_queue(monkeypatch):
    # Focused integration: close admission, fail old pending/queued work, run a
    # fresh job on fresh queues and enforce one restart. Max 8s, no GPU.
    class FakeQueue(queue.Queue):
        closed = False
        def cancel_join_thread(self): pass
        def close(self): self.closed = True
    class FakeProcess:
        def __init__(self, alive=False): self.alive = alive
        def is_alive(self): return self.alive
        def start(self): self.alive = True
        def join(self, timeout): pass
    class FakeContext:
        def __init__(self): self.processes = []
        def Queue(self): return FakeQueue()
        def Process(self, **kwargs):
            process = FakeProcess()
            self.processes.append(process)
            return process
    ctx, old = FakeContext(), FakeProcess()
    old_request, old_response = FakeQueue(), FakeQueue()
    old_request.put({'request_id': 'must-not-replay'})
    executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(main.app.state, 'model_process', old, raising=False)
    monkeypatch.setattr(main.app.state, 'request_queue', old_request, raising=False)
    monkeypatch.setattr(main.app.state, 'response_queue', old_response, raising=False)
    monkeypatch.setattr(main.app.state, 'executor', executor, raising=False)
    monkeypatch.setattr(main.app.state, 'stop_event', threading.Event(), raising=False)
    monkeypatch.setattr(main.app.state, 'lifecycle_stop', asyncio.Event(), raising=False)
    monkeypatch.setattr(main.app.state, 'model_usage', {}, raising=False)
    monkeypatch.setattr(main.app.state, 'worker_generation', 0, raising=False)
    monkeypatch.setattr(main.app.state, 'worker_restarts', 0, raising=False)
    monkeypatch.setattr(main, '_get_cached_result', lambda key: None)
    monkeypatch.setattr(main, 'pending_results', {})
    monkeypatch.setattr(main, 'pending_streams', {})
    future = asyncio.get_running_loop().create_future()
    main.pending_results['old'] = future
    stream = asyncio.Queue()
    main.pending_streams['old-stream'] = stream
    retirement = asyncio.Event()
    monkeypatch.setattr(main.app.state, 'response_listener_retire', retirement, raising=False)
    listener = asyncio.create_task(main.response_listener(None, old_response, retirement))
    monkeypatch.setattr(main.app.state, 'response_listener_task', listener, raising=False)
    supervisor = None
    try:
        settings = {'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False, 'desired_diarize': False}
        store = main._job_store()
        queued = store.create('owner', settings)
        store.enqueue('owner', queued['id'], 1)
        async with main._job_worker_slot:
            main._launch_job('owner', queued)
            supervisor = asyncio.create_task(main._worker_supervisor(ctx))
            for _ in range(100):
                if not main.app.state.worker_accepting: break
                await asyncio.sleep(.01)
            assert not main._worker_accepts_requests()
            for _ in range(200):
                if ctx.processes and main.app.state.worker_accepting:
                    break
                await asyncio.sleep(.02)
            assert len(ctx.processes) == 1 and main.app.state.worker_accepting
            assert old_request.closed and old_response.closed
            assert main.app.state.request_queue is not old_request
            assert main.app.state.request_queue.empty()
            with pytest.raises(main.WorkerStopped):
                await future
            assert 'stopped' in (await stream.get())['error']
            assert await stream.get() is None
        await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 3)
        assert store.get('owner', queued['id'])['status'] == 'error'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://testserver') as client:
            token = (await client.post('/web/session', headers={'origin': 'http://testserver'})).json()['token']
            async def mutate(method, path, **kwargs):
                nonlocal token
                response = await client.request(method, path, headers={'origin': 'http://testserver', 'x-whisper-csrf': token}, **kwargs)
                token = response.headers.get('x-whisper-next-csrf', token)
                return response
            job_id = (await mutate('POST', '/web/jobs', json={'filename': 'x', 'size': 3, 'model': 'small', 'diarize': False})).json()['id']
            await mutate('PUT', f'/web/jobs/{job_id}/upload?offset=0', content=b'abc')
            assert (await mutate('POST', f'/web/jobs/{job_id}/start')).status_code == 200
            for _ in range(100):
                if not main.app.state.request_queue.empty(): break
                await asyncio.sleep(.01)
            request = main.app.state.request_queue.get_nowait()
            assert request['request_id'] != 'must-not-replay'
            main.app.state.response_queue.put({'request_id': request['request_id'], 'result': {'text': 'fresh'}, 'final': True})
            await asyncio.wait_for(asyncio.gather(*list(main._job_tasks)), 3)
            assert (await client.get(f'/web/jobs/{job_id}')).json()['result']['text'] == 'fresh'
        ctx.processes[0].alive = False
        await asyncio.wait_for(supervisor, 3)
        assert len(ctx.processes) == 1 and not main.app.state.worker_accepting
    finally:
        main.app.state.lifecycle_stop.set()
        if supervisor is not None:
            supervisor.cancel()
            await asyncio.gather(supervisor, return_exceptions=True)
        current = main.app.state.response_listener_task
        current.cancel()
        await asyncio.gather(current, return_exceptions=True)
        executor.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('past_deadline', [False, True])
async def test_supervisor_drains_inflight_read_before_replacement(monkeypatch, past_deadline):
    # Focused integration: a read started before retirement holds a final reply;
    # both normal and streaming responses must be delivered. Max 6s; no GPU.
    entered, release = threading.Event(), threading.Event()
    class FakeQueue(queue.Queue):
        def cancel_join_thread(self): pass
        def close(self): pass
    class HeldQueue(FakeQueue):
        first = True
        def get(self, block=True, timeout=None):
            item = super().get(block, timeout)
            if self.first:
                self.first = False
                entered.set()
                assert release.wait(6)
            return item
    class Process:
        def __init__(self, alive=False): self.alive = alive
        def is_alive(self): return self.alive
        def start(self): self.alive = True
        def join(self, timeout): pass
    class Context:
        def Queue(self): return FakeQueue()
        def Process(self, **kwargs): return Process()
    responses, requests = HeldQueue(), FakeQueue()
    responses.put({'request_id': 'normal', 'result': {'text': 'late'}, 'final': True})
    responses.put({'request_id': 'stream', 'segment': {'text': 'segment'}})
    responses.put({'request_id': 'stream', 'result': {'text': 'complete'}, 'final': True})
    executor = ThreadPoolExecutor(max_workers=1)
    retirement = asyncio.Event()
    monkeypatch.setattr(main.app.state, 'model_process', Process(), raising=False)
    monkeypatch.setattr(main.app.state, 'request_queue', requests, raising=False)
    monkeypatch.setattr(main.app.state, 'response_queue', responses, raising=False)
    monkeypatch.setattr(main.app.state, 'executor', executor, raising=False)
    monkeypatch.setattr(main.app.state, 'stop_event', threading.Event(), raising=False)
    monkeypatch.setattr(main.app.state, 'lifecycle_stop', asyncio.Event(), raising=False)
    monkeypatch.setattr(main.app.state, 'model_usage', {}, raising=False)
    monkeypatch.setattr(main.app.state, 'worker_generation', 0, raising=False)
    monkeypatch.setattr(main.app.state, 'worker_restarts', 0, raising=False)
    monkeypatch.setattr(main.app.state, 'response_listener_retire', retirement, raising=False)
    future = asyncio.get_running_loop().create_future()
    stream = asyncio.Queue()
    monkeypatch.setattr(main, 'pending_results', {'normal': future})
    monkeypatch.setattr(main, 'pending_streams', {'stream': stream})
    listener = asyncio.create_task(main.response_listener(None, responses, retirement))
    monkeypatch.setattr(main.app.state, 'response_listener_task', listener, raising=False)
    supervisor = asyncio.create_task(main._worker_supervisor(Context()))
    try:
        for _ in range(200):
            if retirement.is_set(): break
            await asyncio.sleep(.01)
        assert entered.is_set() and retirement.is_set()
        assert not listener.done() and not future.done()
        if past_deadline:
            await asyncio.wait_for(supervisor, 4)
            assert not main.app.state.worker_accepting and not future.done()
        release.set()
        assert (await asyncio.wait_for(future, 2))['text'] == 'late'
        assert (await asyncio.wait_for(stream.get(), 2))['segment']['text'] == 'segment'
        assert (await asyncio.wait_for(stream.get(), 2))['result']['text'] == 'complete'
        assert await asyncio.wait_for(stream.get(), 2) is None
        if not past_deadline:
            for _ in range(200):
                if main.app.state.worker_accepting: break
                await asyncio.sleep(.01)
            assert main.app.state.worker_accepting
        else:
            assert not main.app.state.worker_accepting
    finally:
        release.set()
        main.app.state.lifecycle_stop.set()
        supervisor.cancel()
        await asyncio.gather(supervisor, return_exceptions=True)
        current = main.app.state.response_listener_task
        current.cancel()
        await asyncio.gather(current, return_exceptions=True)
        executor.shutdown(wait=True)
