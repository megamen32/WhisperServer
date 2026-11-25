"""Expire empty browser identities without losing durable history."""
import sqlite3
import time

import httpx
import pytest

import main
from webui_jobs import JobStore, OwnerRevoked


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "WEBUI_JOB_DIR", tmp_path / "jobs")
    monkeypatch.setattr(main, "_identity_cleanup_next", 0.0, raising=False)
    main._webui_sessions.clear()


@pytest.mark.asyncio
async def test_new_session_prunes_expired_empty_identity_and_auth_rows(monkeypatch):
    transport = httpx.ASGITransport(app=main.app)
    old_sessions, old_owners = [], []
    for _ in range(3):
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            assert (await client.post("/web/session", headers={"origin": "http://testserver"})).status_code == 200
            sid = client.cookies.get(main.WEBUI_SESSION_COOKIE)
            old_sessions.append(sid)
            old_owners.append(main._job_store().session_owner(sid))
    with sqlite3.connect(main.WEBUI_JOB_DIR / "sessions.sqlite") as db:
        db.execute("UPDATE sessions SET expires=?", (time.time()-1,))
        db.execute("UPDATE csrf_tokens SET expires=?", (time.time()-1,))
    monkeypatch.setattr(main, "_identity_cleanup_next", 0.0, raising=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as fresh:
        assert (await fresh.post("/web/session", headers={"origin": "http://testserver"})).status_code == 200
        assert (await fresh.get("/web/identity")).json()["job_count"] == 0
    with sqlite3.connect(main._job_store().path) as db:
        remaining = {row[0] for row in db.execute("SELECT id FROM webui_owners")}
        mappings = {row[0] for row in db.execute("SELECT session_id FROM webui_session_owners")}
    assert not remaining.intersection(old_owners), "expired empty owners accumulate indefinitely"
    assert not mappings.intersection(old_sessions)
    with sqlite3.connect(main.WEBUI_JOB_DIR / "sessions.sqlite") as db:
        assert not {row[0] for row in db.execute("SELECT id FROM sessions")}.intersection(old_sessions)
        assert not {row[0] for row in db.execute("SELECT session_id FROM csrf_tokens")}.intersection(old_sessions)


def test_production_create_rejects_deleted_owner(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite")
    settings = {"filename": "audio.wav", "size": 3, "model": "small", "diarize": False}
    with pytest.raises(OwnerRevoked):
        main._create_idempotent_job(store, "deleted-owner", "delayed-request", settings)


def _identity():
    sid, _ = main._create_webui_session()
    return sid, main._job_store().session_owner(sid)


def test_cleanup_preserves_history_keys_and_unexpired_revocation():
    store = main._job_store()
    # Ensure the authentication schema exists before attached cleanup.
    with main._session_db():
        pass
    history_sid, history_owner = _identity()
    key_sid, key_owner = _identity()
    live_sid, live_owner = _identity()
    revoked_sid, revoked_owner = _identity()
    settings = {"filename": "saved.wav", "size": 3, "model": "small", "diarize": False}
    job = store.create(history_owner, settings)
    key = store.issue_recovery_key(key_owner)
    now = time.time()
    with sqlite3.connect(main.WEBUI_JOB_DIR / "sessions.sqlite") as db:
        db.execute("UPDATE sessions SET expires=? WHERE id IN (?,?)",
                   (now-1, history_sid, key_sid))
        db.execute("UPDATE sessions SET expires=? WHERE id=?", (now+100, revoked_sid))
        db.execute("UPDATE sessions SET expires=? WHERE id=?", (now+1000, live_sid))
    with store._transaction() as db:
        db.execute("UPDATE webui_owners SET retired=1 WHERE id=?", (revoked_owner,))
        db.execute("UPDATE webui_session_owners SET revoked=1 WHERE session_id=?", (revoked_sid,))
    store.prune_expired_sessions(main.WEBUI_JOB_DIR / "sessions.sqlite", now)
    assert store.get(history_owner, job["id"])["id"] == job["id"]
    assert store.identity(key_owner)["has_recovery_key"]
    assert store.session_owner(live_sid) == live_owner
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT retired FROM webui_owners WHERE id=?", (revoked_owner,)).fetchone() == (1,)
        assert db.execute("SELECT revoked FROM webui_session_owners WHERE session_id=?", (revoked_sid,)).fetchone() == (1,)
    with pytest.raises(OwnerRevoked):
        store.session_owner(revoked_sid)
    store.prune_expired_sessions(main.WEBUI_JOB_DIR / "sessions.sqlite", now+101)
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT 1 FROM webui_owners WHERE id=?", (revoked_owner,)).fetchone() is None
    # The retained key still recovers the same durable principal after auth expiry.
    recovered = store.recover_owner(live_owner, live_sid, "restored-session", key)
    assert recovered["owner"] == key_owner
    assert store.get(history_owner, job["id"])["owner"] == history_owner


def test_cleanup_limits_each_batch_and_periodic_hook(monkeypatch):
    sid, _ = _identity()
    store = main._job_store()
    now = time.time()
    for index in range(7):
        session = f"expired-{index}"
        store.bind_session(session)
        with main._session_db() as db:
            db.execute("INSERT INTO sessions VALUES(?,?,?)", (session, "token", now-1))
            db.execute("INSERT INTO csrf_tokens VALUES(?,?,?)", (session, "token", now-1))
    counts = store.prune_expired_sessions(main.WEBUI_JOB_DIR / "sessions.sqlite", now, limit=2)
    assert counts == {"mappings": 2, "owners": 2, "tokens": 2, "sessions": 2}
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM webui_owners").fetchone()[0] == 6

    calls = []
    original = store.prune_expired_sessions
    monkeypatch.setattr(store, "prune_expired_sessions",
                        lambda *args, **kwargs: calls.append(args) or original(*args, **kwargs))
    monkeypatch.setattr(main, "_identity_cleanup_next", 0.0)
    _identity()
    _identity()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_recovery_retains_revoked_auth_expiry_until_cleanup():
    transport = httpx.ASGITransport(app=main.app)
    target_sid, target_owner = _identity()
    key = main._job_store().issue_recovery_key(target_owner)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        token = (await client.post("/web/session", headers={"origin": "http://testserver"})).json()["token"]
        old_sid = client.cookies.get(main.WEBUI_SESSION_COOKIE)
        response = await client.post("/web/identity/recover",
            headers={"origin": "http://testserver", "x-whisper-csrf": token},
            json={"recovery_key": key})
        assert response.status_code == 200
        assert main._load_webui_session(old_sid) is None
        with sqlite3.connect(main.WEBUI_JOB_DIR / "sessions.sqlite") as db:
            expiry = db.execute("SELECT expires FROM sessions WHERE id=?", (old_sid,)).fetchone()
            assert expiry and expiry[0] > time.time()
            assert not db.execute("SELECT 1 FROM csrf_tokens WHERE session_id=?", (old_sid,)).fetchone()


@pytest.mark.asyncio
@pytest.mark.parametrize("idempotent", [False, True])
async def test_inflight_http_create_cannot_recreate_expired_owner(idempotent):
    import asyncio
    import json
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        token = (await client.post("/web/session", headers={"origin": "http://testserver"})).json()["token"]
        sid = client.cookies.get(main.WEBUI_SESSION_COOKIE)
        owner = main._job_store().session_owner(sid)
        entered, resume = asyncio.Event(), asyncio.Event()
        payload = {"filename": "late.wav", "size": 3, "model": "small", "diarize": False}
        if idempotent: payload["client_request_id"] = "delayed"
        async def body():
            entered.set()
            await resume.wait()
            yield json.dumps(payload).encode()
        request = asyncio.create_task(client.post("/web/jobs", content=body(), headers={
            "origin": "http://testserver", "x-whisper-csrf": token, "content-type": "application/json"}))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            now = time.time()
            with sqlite3.connect(main.WEBUI_JOB_DIR / "sessions.sqlite") as db:
                db.execute("UPDATE sessions SET expires=? WHERE id=?", (now-1, sid))
            main._job_store().prune_expired_sessions(main.WEBUI_JOB_DIR / "sessions.sqlite", now)
            resume.set()
            response = await request
            assert response.status_code == 403
            assert main._job_store().list(owner) == []
        finally:
            resume.set()
            if not request.done():
                request.cancel()
            await asyncio.gather(request, return_exceptions=True)
