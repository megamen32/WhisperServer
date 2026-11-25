"""Identity recovery and source-only removal use the real owner/CSRF API."""
import json
import sqlite3
import time

import httpx
import pytest

import main


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "WEBUI_JOB_DIR", tmp_path / "jobs")
    main._webui_sessions.clear()
    main._webui_transcripts.clear()


async def session(client):
    response = await client.post("/web/session", headers={"origin": "http://testserver"})
    assert response.status_code == 200
    return response.json()["token"]


async def mutate(client, method, path, **kwargs):
    token = await session(client)
    return await client.request(method, path, headers={
        "origin": "http://testserver", "x-whisper-csrf": token}, **kwargs)


async def complete_job(client, text="saved text"):
    response = await mutate(client, "POST", "/web/jobs", json={
        "filename": "audio.wav", "size": 3, "model": "small", "diarize": False})
    assert response.status_code == 200
    job = response.json()
    main._job_spool().append(job["id"], 0, b"abc", 3)
    store = main._job_store()
    store.enqueue(job["owner"], job["id"], 1)
    store.start(job["owner"], job["id"], 1)
    store.complete(job["owner"], job["id"], 1, {
        "text": text, "segments": [{"start": 0, "end": 1, "text": text}]})
    store.put_summary(job["owner"], job["id"], 1, "latest", {"summary": "saved summary", "attempt": 1})
    return job


@pytest.mark.asyncio
async def test_lost_cookie_recovery_restores_history_and_rotates_current_session():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as first:
        job = await complete_job(first)
        old_sid = first.cookies.get(main.WEBUI_SESSION_COOKIE)
        issued = await mutate(first, "POST", "/web/identity/recovery-key", json={})
        assert issued.status_code == 200
        recovery_key = issued.json()["recovery_key"]
        with sqlite3.connect(main._job_store().path) as db:
            assert not any(recovery_key in str(row) for row in db.execute("SELECT * FROM webui_owners"))

        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as second:
            await session(second)
            anonymous_sid = second.cookies.get(main.WEBUI_SESSION_COOKIE)
            assert (await second.get("/web/jobs")).json()["jobs"] == []
            assert (await second.get(f"/web/jobs/{job['id']}")).status_code == 404
            recovered = await mutate(second, "POST", "/web/identity/recover", json={"recovery_key": recovery_key})
            assert recovered.status_code == 200
            restored = (await second.get(f"/web/jobs/{job['id']}")).json()
            assert restored["result"]["text"] == "saved text"
            assert restored["summaries"]["latest"]["summary"] == "saved summary"
            assert second.cookies.get(main.WEBUI_SESSION_COOKIE) not in (old_sid, anonymous_sid)
            assert main._load_webui_session(anonymous_sid) is None
            # Other sessions already belonging to the target owner remain valid.
            assert (await first.get(f"/web/jobs/{job['id']}")).status_code == 200


@pytest.mark.asyncio
async def test_recovery_requires_explicit_merge_and_preserves_both_histories():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as original, \
               httpx.AsyncClient(transport=transport, base_url="http://testserver") as current:
        original_job = await complete_job(original, "original")
        issued = await mutate(original, "POST", "/web/identity/recovery-key", json={})
        assert issued.status_code == 200
        key = issued.json()["recovery_key"]
        current_job = await complete_job(current, "current")
        current_sid = current.cookies.get(main.WEBUI_SESSION_COOKIE)
        refused = await mutate(current, "POST", "/web/identity/recover", json={"recovery_key": key})
        assert refused.status_code == 409
        assert (await current.get(f"/web/jobs/{current_job['id']}")).status_code == 200
        merged = await mutate(current, "POST", "/web/identity/recover",
                              json={"recovery_key": key, "merge_current": True})
        assert merged.status_code == 200
        ids = {job["id"] for job in (await current.get("/web/jobs")).json()["jobs"]}
        assert ids == {original_job["id"], current_job["id"]}
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as stale:
            stale.cookies.set(main.WEBUI_SESSION_COOKIE, current_sid)
            assert (await stale.get("/web/jobs")).status_code == 403


@pytest.mark.asyncio
async def test_remove_source_keeps_transcript_summary_and_rejects_rerun():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://testserver") as client:
        job = await complete_job(client)
        removed = await mutate(client, "DELETE", f"/web/jobs/{job['id']}/source")
        assert removed.status_code == 200
        assert main._job_spool().offset(job["id"]) == 0
        saved = (await client.get(f"/web/jobs/{job['id']}")).json()
        assert saved["result"]["text"] == "saved text"
        assert saved["summaries"]["latest"]["summary"] == "saved summary"
        assert saved["source_available"] is False
        retry = await mutate(client, "PATCH", f"/web/jobs/{job['id']}", json={"diarize": True})
        assert retry.status_code == 409
        assert (await client.get(f"/web/jobs/{job['id']}")).json()["status"] == "done"


@pytest.mark.asyncio
async def test_legacy_database_and_expired_session_recover_without_losing_text(tmp_path):
    # This is the pre-owner schema, with a saved full result and a summary.
    legacy_sid, _ = main._create_webui_session(bind_owner=False)
    database = main.WEBUI_JOB_DIR / "jobs.sqlite"
    payload = {"text": "legacy text", "segments": [{"start": 0, "end": 1, "text": "legacy text"}]}
    settings = {"filename": "legacy.wav", "size": 3, "model": "small", "diarize": False}
    with sqlite3.connect(database) as db:
        db.executescript("""
            CREATE TABLE jobs(id TEXT PRIMARY KEY,owner TEXT NOT NULL,status TEXT NOT NULL,
                attempt INTEGER NOT NULL,diarization TEXT NOT NULL,result TEXT,error TEXT,
                created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
            CREATE TABLE summaries(job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                name TEXT NOT NULL,value TEXT NOT NULL,PRIMARY KEY(job_id,name));
        """)
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,NULL,?,?)",
                   ("legacy", legacy_sid, "done", 1, json.dumps(settings), json.dumps(payload), "before", "before"))
        db.execute("INSERT INTO summaries VALUES(?,?,?)",
                   ("legacy", "latest", json.dumps({"summary": "legacy summary"})))
    main._job_spool().append("legacy", 0, b"abc", 3)
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        client.cookies.set(main.WEBUI_SESSION_COOKIE, legacy_sid, domain="testserver.local", path="/")
        migrated = (await client.get("/web/jobs/legacy")).json()
        assert migrated["owner"] == legacy_sid
        assert migrated["result"] == payload
        assert migrated["summaries"]["latest"]["summary"] == "legacy summary"
        stream = await client.get("/web/jobs/legacy/events?protocol=delta")
        assert stream.status_code == 200
        assert "legacy text" in stream.text
        issued = await mutate(client, "POST", "/web/identity/recovery-key", json={})
        key = issued.json()["recovery_key"]
        with main._session_db() as db:
            db.execute("UPDATE sessions SET expires=? WHERE id=?", (time.time() - 1, legacy_sid))
        assert (await client.get("/web/jobs/legacy")).status_code == 403
        await session(client)
        fresh_sid = client.cookies.get(main.WEBUI_SESSION_COOKIE)
        fresh_owner = main._load_webui_session(fresh_sid)["owner_id"]
        assert fresh_owner not in (fresh_sid, legacy_sid)
        assert (await client.get("/web/jobs")).json()["jobs"] == []
        restored = await mutate(client, "POST", "/web/identity/recover", json={"recovery_key": key})
        assert restored.status_code == 200
        job = (await client.get("/web/jobs/legacy")).json()
        assert job["result"] == payload
        assert job["summaries"]["latest"]["summary"] == "legacy summary"
        assert main._job_spool().offset("legacy") == 3


@pytest.mark.asyncio
async def test_identity_csrf_rotation_and_cross_owner_isolation():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as owner, \
               httpx.AsyncClient(transport=transport, base_url="http://testserver") as outsider:
        job = await complete_job(owner)
        token = await session(owner)
        assert (await owner.post("/web/identity/recovery-key", json={})).status_code == 403
        wrong_origin = await owner.post("/web/identity/recovery-key", json={}, headers={
            "origin": "https://foreign.invalid", "x-whisper-csrf": token})
        assert wrong_origin.status_code == 403
        first = await mutate(owner, "POST", "/web/identity/recovery-key", json={})
        first_key = first.json()["recovery_key"]
        assert first.headers["cache-control"] == "no-store"
        info = await owner.get("/web/identity")
        assert set(info.json()) == {"has_recovery_key", "job_count"}
        assert first_key not in info.text
        assert (await mutate(owner, "POST", "/web/identity/recovery-key", json={})).status_code == 409
        second = await mutate(owner, "POST", "/web/identity/recovery-key", json={"replace": True})
        second_key = second.json()["recovery_key"]
        assert first_key != second_key
        await session(outsider)
        for path in (f"/web/jobs/{job['id']}", f"/web/jobs/{job['id']}/events"):
            assert (await outsider.get(path)).status_code == 404
        assert (await mutate(outsider, "DELETE", f"/web/jobs/{job['id']}/source")).status_code == 404
        assert (await mutate(outsider, "POST", "/web/summarize",
                             json={"transcript_id": job["id"]})).status_code == 404
        rejected = await mutate(outsider, "POST", "/web/identity/recover", json={"recovery_key": first_key})
        assert rejected.status_code == 400
        assert first_key not in rejected.text
        recovered = await mutate(outsider, "POST", "/web/identity/recover", json={"recovery_key": second_key})
        assert recovered.status_code == 200
        # The token returned with the rotated session is immediately usable once.
        next_token = recovered.headers["x-whisper-next-csrf"]
        metadata = {"filename": "next.wav", "size": 1, "model": "small", "diarize": False}
        headers = {"origin": "http://testserver", "x-whisper-csrf": next_token}
        assert (await outsider.post("/web/jobs", json=metadata, headers=headers)).status_code == 200
        assert (await outsider.post("/web/jobs", json=metadata, headers=headers)).status_code == 403


@pytest.mark.asyncio
async def test_active_job_blocks_merge_then_merge_revokes_source_key_and_sessions():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as target, \
               httpx.AsyncClient(transport=transport, base_url="http://testserver") as source, \
               httpx.AsyncClient(transport=transport, base_url="http://testserver") as other_source:
        target_job = await complete_job(target, "target")
        target_key = (await mutate(target, "POST", "/web/identity/recovery-key", json={})).json()["recovery_key"]
        source_job = await complete_job(source, "source")
        source_owner = source_job["owner"]
        source_key = (await mutate(source, "POST", "/web/identity/recovery-key", json={})).json()["recovery_key"]
        assert (await mutate(other_source, "POST", "/web/identity/recover",
                             json={"recovery_key": source_key})).status_code == 200
        pending = await mutate(source, "POST", "/web/jobs", json={
            "filename": "active.wav", "size": 1, "model": "small", "diarize": False})
        pending_job = pending.json()
        refused = await mutate(source, "POST", "/web/identity/recover",
                               json={"recovery_key": target_key, "merge_current": True})
        assert refused.status_code == 409 and refused.json()["detail"]["code"] == "active_jobs"
        assert (await other_source.get(f"/web/jobs/{source_job['id']}")).status_code == 200
        assert (await target.get(f"/web/jobs/{source_job['id']}")).status_code == 404
        cancelled = await mutate(source, "POST", f"/web/jobs/{pending_job['id']}/cancel")
        assert cancelled.status_code == 200
        merged = await mutate(source, "POST", "/web/identity/recover",
                              json={"recovery_key": target_key, "merge_current": True})
        assert merged.status_code == 200 and merged.json()["merged_jobs"] == 2
        assert (await other_source.get("/web/jobs")).status_code == 403
        source_key_rejected = await mutate(other_source, "POST", "/web/identity/recover",
                                          json={"recovery_key": source_key})
        assert source_key_rejected.status_code == 400
        # Requests which captured the old owner before merge must not strand new jobs.
        store = main._job_store()
        with pytest.raises(main.OwnerRevoked):
            store.create(source_owner, {})
        with pytest.raises(main.OwnerRevoked):
            main._create_idempotent_job(store, source_owner, "old-request", {})
        assert (await target.get(f"/web/jobs/{target_job['id']}")).status_code == 200
        assert (await target.get(f"/web/jobs/{source_job['id']}")).status_code == 200


@pytest.mark.asyncio
async def test_source_removal_is_idempotent_and_does_not_affect_other_audio():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://testserver") as client:
        removed = await complete_job(client, "remove audio")
        retained = await complete_job(client, "retain audio")
        uploading = await mutate(client, "POST", "/web/jobs", json={
            "filename": "upload.wav", "size": 3, "model": "small", "diarize": False})
        assert (await mutate(client, "DELETE", f"/web/jobs/{uploading.json()['id']}/source")).status_code == 409
        for _ in range(2):
            assert (await mutate(client, "DELETE", f"/web/jobs/{removed['id']}/source")).status_code == 200
        assert main._job_spool().offset(retained["id"]) == 3
        assert len((await client.get("/web/jobs")).json()["jobs"]) == 3
        with pytest.raises(main.InvalidTransition, match="Source audio was removed"):
            main._job_store().retry(removed["owner"], removed["id"], removed["diarization"])
        assert main._job_store().get(removed["owner"], removed["id"])["summaries"]["latest"]["summary"] == "saved summary"


@pytest.mark.asyncio
async def test_uncached_summary_after_recovery_is_saved_to_durable_owner(monkeypatch):
    async def summarize(*args):
        return '{"choices":[{"message":{"content":"new summary"}}],"model":"fixture-model"}'

    monkeypatch.setattr(main, "_summary_open", summarize)
    monkeypatch.setattr(main, "_omniroute_api_key", lambda: "fixture-not-a-real-key")
    main._webui_summary_calls.clear()
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as owner, \
               httpx.AsyncClient(transport=transport, base_url="http://testserver") as restored:
        job = await complete_job(owner)
        key = (await mutate(owner, "POST", "/web/identity/recovery-key", json={})).json()["recovery_key"]
        assert (await mutate(restored, "POST", "/web/identity/recover",
                             json={"recovery_key": key})).status_code == 200
        summary = await mutate(restored, "POST", "/web/summarize",
                               json={"transcript_id": job["id"], "force_refresh": True})
        assert summary.status_code == 200
        saved = (await restored.get(f"/web/jobs/{job['id']}")).json()["summaries"]["latest"]
        assert saved["summary"] == "new summary"


@pytest.mark.asyncio
async def test_failed_source_unlink_can_be_retried_without_losing_text(monkeypatch):
    transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        job = await complete_job(client)
        spool = main._job_spool()
        remove = spool.remove
        failed = False

        def fail_once(*args):
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("temporary fixture filesystem failure")
            return remove(*args)

        monkeypatch.setattr(spool, "remove", fail_once)
        response = await mutate(client, "DELETE", f"/web/jobs/{job['id']}/source")
        assert response.status_code == 503
        saved = (await client.get(f"/web/jobs/{job['id']}")).json()
        assert saved["source_available"] is False and saved["source_removal_pending"] is True
        assert saved["result"]["text"] == "saved text"
        retried = await mutate(client, "DELETE", f"/web/jobs/{job['id']}/source")
        assert retried.status_code == 200 and retried.json()["source_removal_pending"] is False
        assert spool.offset(job["id"]) == 0


@pytest.mark.asyncio
async def test_source_removal_also_evicts_unborrowed_prepared_audio(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    from prepared_audio import PreparedAudioCache

    pcm_root = tmp_path / "pcm"
    monkeypatch.setenv("WHISPER_PCM_DIR", str(pcm_root))
    cache = PreparedAudioCache(pcm_root / "cache-v1", min_free_bytes=0)
    audio_hash = hashlib.sha256(b"abc").hexdigest()
    idle = cache.borrow(audio_hash, {"decoder": "idle"}, lambda: np.zeros(20, dtype=np.float32))
    idle_path = idle.path
    idle.close()
    active = cache.borrow(audio_hash, {"decoder": "active"}, lambda: np.zeros(20, dtype=np.float32))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://testserver") as client:
            job = await complete_job(client)
            removed = await mutate(client, "DELETE", f"/web/jobs/{job['id']}/source")
            assert removed.status_code == 200
            assert not idle_path.exists(), "deleted source left an unused decoded audio copy"
            assert active.path.exists(), "source deletion invalidated another active audio borrower"
            saved = (await client.get(f"/web/jobs/{job['id']}")).json()
            assert saved["result"]["text"] == "saved text"
            assert saved["summaries"]["latest"]["summary"] == "saved summary"
    finally:
        active.close()
