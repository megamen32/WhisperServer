"""Small, bounded SQLite registry for Web UI jobs.

Each operation opens its own connection, so instances can be used by different
threads or processes. Owner identifiers must come from the authenticated caller,
not from untrusted request data. Payloads are JSON-compatible values.
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone


class JobNotFound(LookupError):
    """The job does not exist or is not owned by the caller."""


class InvalidTransition(ValueError):
    """The requested state transition is no longer valid."""


class CapacityExceeded(ValueError):
    """A configured storage limit would be exceeded."""


def _now():
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """Persistent job store; limits apply to the whole database, not per process.

    Use a file-backed database: each call uses a separate SQLite connection.
    max_payload_bytes limits each JSON value and max_summaries limits each job.
    Terminal jobs are explicitly deleted by the owner to reclaim capacity.
    """

    def __init__(self, path, *, max_jobs=1000, max_payload_bytes=1_000_000,
                 max_summaries=20, timeout=5):
        if str(path) == ':memory:':
            raise ValueError('a file-backed database is required')
        if min(max_jobs, max_payload_bytes, max_summaries) < 1:
            raise ValueError('limits must be positive')
        self.path = str(path)
        self.max_jobs = max_jobs
        self.max_payload_bytes = max_payload_bytes
        self.max_summaries = max_summaries
        self.timeout = timeout
        with self._transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL,
                attempt INTEGER NOT NULL, diarization TEXT NOT NULL,
                result TEXT, error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )''')
            db.execute('''CREATE TABLE IF NOT EXISTS summaries (
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                name TEXT NOT NULL, value TEXT NOT NULL,
                PRIMARY KEY (job_id, name)
            )''')
            db.execute('CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner, created_at)')

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=self.timeout, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute('PRAGMA foreign_keys = ON')
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _json(self, value):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        if len(data.encode('utf-8')) > self.max_payload_bytes:
            raise CapacityExceeded('JSON payload exceeds max_payload_bytes')
        return data

    @staticmethod
    def _identity(value, name):
        if not isinstance(value, str) or not value or len(value.encode('utf-8')) > 512:
            raise ValueError(f'{name} must be a nonempty string of at most 512 bytes')

    @staticmethod
    def _owned(db, owner, job_id):
        row = db.execute('SELECT * FROM jobs WHERE id = ? AND owner = ?', (job_id, owner)).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        return row

    @staticmethod
    def _record(db, row):
        data = dict(row)
        data['diarization'] = json.loads(data['diarization'])
        data['result'] = json.loads(data['result']) if data['result'] is not None else None
        data['summaries'] = {r['name']: json.loads(r['value']) for r in db.execute(
            'SELECT name, value FROM summaries WHERE job_id = ? ORDER BY name', (row['id'],))}
        return data

    def create(self, owner, diarization):
        self._identity(owner, 'owner')
        job_id = uuid.uuid4().hex
        settings = self._json(diarization)
        now = _now()
        with self._transaction() as db:
            if db.execute('SELECT count(*) FROM jobs').fetchone()[0] >= self.max_jobs:
                raise CapacityExceeded('max_jobs reached')
            db.execute('INSERT INTO jobs VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?)',
                       (job_id, owner, 'uploading', 1, settings, now, now))
            return self._record(db, self._owned(db, owner, job_id))

    def get(self, owner, job_id):
        with self._transaction() as db:
            return self._record(db, self._owned(db, owner, job_id))

    def list(self, owner):
        self._identity(owner, 'owner')
        with self._transaction() as db:
            rows = db.execute('SELECT * FROM jobs WHERE owner = ? ORDER BY created_at, id', (owner,)).fetchall()
            return [self._record(db, row) for row in rows]

    def _transition(self, owner, job_id, attempt, allowed, status, *, result=None, error=None):
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['attempt'] != attempt or row['status'] not in allowed:
                raise InvalidTransition('stale attempt or invalid state')
            db.execute('UPDATE jobs SET status=?, result=?, error=?, updated_at=? WHERE id=?',
                       (status, result, error, _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def enqueue(self, owner, job_id, attempt):
        return self._transition(owner, job_id, attempt, ('uploading',), 'queued')

    def start(self, owner, job_id, attempt):
        return self._transition(owner, job_id, attempt, ('queued',), 'processing')

    def request_cancel(self, owner, job_id, attempt):
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['attempt'] != attempt or row['status'] not in ('uploading', 'queued', 'processing', 'cancel_requested'):
                raise InvalidTransition('stale attempt or invalid state')
            status = 'cancel_requested' if row['status'] in ('processing', 'cancel_requested') else 'cancelled'
            db.execute('UPDATE jobs SET status=?, updated_at=? WHERE id=?', (status, _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def finish_cancel(self, owner, job_id, attempt):
        return self._transition(owner, job_id, attempt, ('cancel_requested',), 'cancelled')

    def complete(self, owner, job_id, attempt, result):
        payload = self._json(result)
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            settings = json.loads(row['diarization'])
            if (row['attempt'] != attempt or row['status'] != 'processing' or
                    settings.get('desired_diarize', settings.get('diarize')) != settings.get('diarize')):
                raise InvalidTransition('stale attempt or changed settings')
            db.execute('UPDATE jobs SET status=?, result=?, error=NULL, updated_at=? WHERE id=?',
                       ('done', payload, _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def set_diarization(self, owner, job_id, attempt, settings):
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['attempt'] != attempt or row['status'] not in ('uploading', 'queued', 'processing', 'done', 'error'):
                raise InvalidTransition('stale attempt or invalid state')
            desired = settings['desired_diarize']
            status, next_attempt = row['status'], attempt
            if status in ('uploading', 'queued'):
                settings['diarize'] = desired
            elif status in ('done', 'error') and settings['diarize'] != desired:
                settings['diarize'] = desired
                status, next_attempt = 'queued', attempt + 1
                db.execute('DELETE FROM summaries WHERE job_id=?', (job_id,))
            db.execute('UPDATE jobs SET diarization=?, status=?, attempt=?, result=?, error=NULL, updated_at=? WHERE id=?',
                       (self._json(settings), status, next_attempt, row['result'] if status == 'done' else None, _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def retry_if_diarization_changed(self, owner, job_id, attempt):
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            settings = json.loads(row['diarization'])
            desired = settings.get('desired_diarize', settings.get('diarize'))
            if row['attempt'] != attempt or row['status'] not in ('processing', 'error', 'done') or desired == settings.get('diarize'):
                return None
            settings['diarize'] = desired
            db.execute('DELETE FROM summaries WHERE job_id=?', (job_id,))
            db.execute('UPDATE jobs SET status=?, attempt=?, diarization=?, result=NULL, error=NULL, updated_at=? WHERE id=?',
                       ('queued', attempt + 1, self._json(settings), _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def recover_interrupted(self):
        with self._transaction() as db:
            rows = db.execute("SELECT * FROM jobs WHERE status IN ('queued', 'processing', 'cancel_requested', 'error')").fetchall()
            result = []
            for row in rows:
                if row['status'] == 'cancel_requested':
                    db.execute("UPDATE jobs SET status='cancelled', updated_at=? WHERE id=?", (_now(), row['id']))
                    continue
                settings = json.loads(row['diarization'])
                if row['status'] == 'error' and settings.get('desired_diarize', settings.get('diarize')) == settings.get('diarize'):
                    continue
                settings['diarize'] = settings.get('desired_diarize', settings.get('diarize'))
                db.execute("UPDATE jobs SET status='queued', attempt=attempt+1, diarization=?, updated_at=? WHERE id=?",
                           (self._json(settings), _now(), row['id']))
                result.append(self._record(db, self._owned(db, row['owner'], row['id'])))
            return result

    def fail(self, owner, job_id, attempt, error):
        if not isinstance(error, str):
            raise ValueError('error must be a string')
        if len(error.encode('utf-8')) > self.max_payload_bytes:
            raise CapacityExceeded('error exceeds max_payload_bytes')
        return self._transition(owner, job_id, attempt, ('uploading', 'queued', 'processing'), 'error', error=error)

    def retry(self, owner, job_id, diarization):
        settings = self._json(diarization)
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['status'] not in ('done', 'error', 'cancelled'):
                raise InvalidTransition('only terminal jobs can be retried')
            db.execute('DELETE FROM summaries WHERE job_id=?', (job_id,))
            db.execute('''UPDATE jobs SET attempt=attempt+1, status='queued', diarization=?,
                          result=NULL, error=NULL, updated_at=? WHERE id=?''', (settings, _now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def put_summary(self, owner, job_id, attempt, name, value):
        self._identity(name, 'name')
        payload = self._json(value)
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['attempt'] != attempt or row['status'] != 'done':
                raise InvalidTransition('summary requires the current completed attempt')
            exists = db.execute('SELECT 1 FROM summaries WHERE job_id=? AND name=?', (job_id, name)).fetchone()
            if not exists and db.execute('SELECT count(*) FROM summaries WHERE job_id=?', (job_id,)).fetchone()[0] >= self.max_summaries:
                raise CapacityExceeded('max_summaries reached')
            db.execute('''INSERT INTO summaries(job_id,name,value) VALUES (?,?,?)
                          ON CONFLICT(job_id,name) DO UPDATE SET value=excluded.value''', (job_id, name, payload))
            db.execute('UPDATE jobs SET updated_at=? WHERE id=?', (_now(), job_id))
            return self._record(db, self._owned(db, owner, job_id))

    def delete(self, owner, job_id):
        with self._transaction() as db:
            row = self._owned(db, owner, job_id)
            if row['status'] not in ('done', 'error', 'cancelled'):
                raise InvalidTransition('only terminal jobs can be deleted')
            db.execute('DELETE FROM jobs WHERE id=?', (job_id,))
